# utils_loop.py
# -*- coding: utf-8 -*-
#只加了工具函数，整体的解析逻辑没改，还是采用之前的api提取datalink

import os, re, json, time, math, asyncio, logging, random, threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random, requests
from datetime import timedelta
import tarfile
import tempfile
from urllib.parse import unquote
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import threading
# ===== 在文件顶部新增（或在 utils 中提供并导入） =====
from transformers import AutoTokenizer
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout, SSLError


# ==== 0 路径与配置 ====
class CONFIG:
    LAST_N_DAYS = 3          # 近一月=30；近一年=365；不用则设为 None
    ARXIV_DATE_FIELD = "lastUpdatedDate"   # 可改为 "lastUpdatedDate"（包含修订更稳） 
    
    BASE_DIR = "/home/kzlab/muse/Savvy/Data_collection"
    MODELS_DIR = f"{BASE_DIR}/models"
    OUTPUT_DIR = f"{BASE_DIR}/output"
    OUT_PAPERS = f"{OUTPUT_DIR}/1_paper"            # 仅元数据备份（按天/类目）
    OUT_PROCESSED = f"{OUTPUT_DIR}/2_paper_processed"
    OUT_PDF = f"{OUTPUT_DIR}/3_pdf"
    OUT_FINAL = f"{OUTPUT_DIR}/final_dataset"       # << 当日最终四列 JSON
    STATE_DIR = f"{BASE_DIR}/state"                 # 拉取与处理状态

    # arXiv 类目
    CATEGORIES = ["cs.IR","cs.DB","cs.AI","cs.CL","cs.CV","cs.MA"]

    # arXiv API（按提交时间倒序）
    ARXIV_API =  "https://export.arxiv.org/api/query"

    # 轮询周期（分钟）
    POLL_MINUTES = 30

    # ======= 并发控制 =======
    # 同时并行处理的“论文数”（下载/GROBID/贴标签在每个任务内顺序执行）
    MAX_PAPER_WORKERS = 1
    # MAX_PAPER_WORKERS =1 #debug用
    
    # LLM 并发数上限（信号量控制，避免被限流）
    MAX_LLM_WORKERS = 2
    # （保留）未直接使用；如需细分 PDF/GROBID 并发，可再引入对应信号量
    MAX_PDF_WORKERS = 2

    # BERT 路径（用于 title+abstract 二分类）
    BERT_TOKENIZER = f"{MODELS_DIR}/fine_tuned_model"
    BERT_CKPT = f"{MODELS_DIR}/checkpoint-1530"
    BERT_THRESHOLD = 0.50

    # GROBID
    GROBID_URL = "http://localhost:8071/api/processFulltextDocument"

    # ===================LLM（阿里灵积 DashScope 兼容 OpenAI） ====================================
    
    # LLM_BASE_URL = "http://127.0.0.1:8000/v1"
    # LLM_MODEL = "QwQ-32B"
    LLM_TEMPERATURE = 0.15
    DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY_ENV"  # 请 export 到环境变量

    # 随机抖动（防止各类目同时打 API）
    JITTER_SEC = (0.0, 2.0)

    # ==== 目标日期与去重策略 ====
    # TARGET_DAY = "2025-09-21"       # 想按系统当天请设为 None；或填 "YYYY-MM-DD"
    TARGET_DAY = None
    IGNORE_SEEN_WHEN_TARGETED = True  # 指定 TARGET_DAY 时是否忽略历史去重
    ARXIV_MAX_RESULTS = 200           # 每类目拉取更充足，避免当天条目被挤出前N
    
    
    # ==== 4  关于LaTex配置部分  ====
    PDF_DIR = f"{OUTPUT_DIR}/3_pdf" 
    OUT_LATEX = f"{OUTPUT_DIR}/4_latex"        # 存放源码包与解析产物
    LATEX_ENABLED = True                       # 总开关
    LATEX_MAX_TARBALL_MB = 50                  # 源码最大体积（安全上限） 
    CTX_APPEND_TO_DATA_DESC = True      # 把窗口上下文并入 dataset_description
    CTX_MAX_CHARS = 2000                # 注入的上下文最长字符数（防止过长）

    
# 全局会话（长连接 + 重试）
_SESSION = requests.Session()
_RETRY = Retry(
    total=4, connect=3, read=3,
    backoff_factor=1.2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["POST"])
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY, pool_connections=20, pool_maxsize=50)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

# ==== 日志 ====
LOG_DIR = os.path.join(CONFIG.BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "realtime.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),           # 继续在控制台打印
        logging.FileHandler(log_path, encoding="utf-8")  # 额外写入文件
    ]    
)
logger = logging.getLogger("realtime")

# ==== 并发相关全局对象 ====
_LLM_SEM = threading.Semaphore(CONFIG.MAX_LLM_WORKERS)  # LLM 并发限流
_FINAL_WRITE_LOCK = threading.Lock()                    # 写 final json 的互斥锁

# ==== 工具 ====
def _post_qwen_with_retry(prompt_json, timeout=(10, 120), tries=3):
    s = llm_client()  # <<< 用带 Authorization 的会话
    url = CONFIG.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    for k in range(1, tries+1):
        try:
            r = s.post(url, json=prompt_json, timeout=timeout)
            if r.status_code == 401:
                logger.error("LLM 401 Unauthorized. 请检查 API key / 模型权限。resp=%s", r.text[:500])
            r.raise_for_status()
            return r.json()
        
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            if k == tries:
                raise
            time.sleep((2 ** k) + random.random())
        except requests.HTTPError as e:
            if k == tries or (e.response is not None and e.response.status_code in (401, 403, 404)):
                # 配置/权限问题没必要重试太多
                raise
            time.sleep((2 ** k) + random.random())

def latex_src_url(arxiv_id: str) -> str:
    """
    arXiv 源码直链（无需 cookies 的常见方式）：
    - 优先尝试 e-print tar 包： https://arxiv.org/e-print/<id>
    - 若失败你也可以降级走 “其他格式页” 去解析，但那要多一次 HTML 请求，这里先不做。
    """
    return f"https://arxiv.org/e-print/{arxiv_id}"

def _human_mb(n_bytes:int)->float:
    return round(n_bytes/1024/1024, 3)

def download_latex_source(arxiv_id: str, save_path: str) -> bool:
    """
    直接 GET e-print tar 包；成功返回 True。
    失败场景：返回 403/404、Content-Type 不是 tar、体积过大等。
    """
    try:
        ensure_dir(os.path.dirname(save_path))
        s = _session()
        s.headers.update({"User-Agent": "DC-realtime-pipeline/1.0 (mailto:1135972876@qq.com)"})
        url = latex_src_url(arxiv_id)
        with s.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                logger.info("No LaTeX source for %s (status=%s)", arxiv_id, r.status_code)
                return False
            # 体积控制（防止异常大包）
            total = 0
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
                    if _human_mb(total) > CONFIG.LATEX_MAX_TARBALL_MB:
                        logger.warning("Latex tar too large: %s > %s MB", _human_mb(total), CONFIG.LATEX_MAX_TARBALL_MB)
                        return False
        return True
    except Exception as e:
        logger.exception("download_latex_source error: %s", e)
        return False

def extract_tar(tar_path: str, out_dir: str) -> bool:
    try:
        ensure_dir(out_dir)
        mode = "r:gz" if tar_path.endswith(".gz") else "r:*"
        with tarfile.open(tar_path, mode) as tf:
            def is_safe(member: tarfile.TarInfo) -> bool:
                # 基础安全：禁止绝对路径/.. 穿越
                name = member.name
                if name.startswith("/") or ".." in name.replace("\\","/"):
                    return False
                return True
            safe_members = [m for m in tf.getmembers() if is_safe(m)]
            tf.extractall(out_dir, members=safe_members)
        return True
    except Exception as e:
        logger.exception("extract_tar error: %s", e)
        return False

_URL_PAT = re.compile(
    r'(?P<url>(?:https?://|ftp://)[^\s<>{}|\^\[\]`\\)"]+)',
    re.IGNORECASE
)

def _iter_text_files(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for nm in files:
            low = nm.lower()
            if low.endswith((".tex", ".bib", ".txt", ".md")):
                yield os.path.join(root, nm)

def parse_links_from_tex_dir(src_dir: str) -> List[str]:
    r"""
    解析目录内的 .tex/.bib 文本，提取 URL（含 \url{} \href{} 中的链接以及裸链）。
    """
    urls = set()
    # 常见 LaTeX 链接形式
    href_pat = re.compile(r'\\href\{([^}]+)\}\{[^}]*\}')
    url_pat  = re.compile(r'\\url\{([^}]+)\}')
    for fp in _iter_text_files(src_dir):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            # \href{url}{text} / \url{url}
            for m in href_pat.finditer(txt):
                urls.add(m.group(1).strip())
            for m in url_pat.finditer(txt):
                urls.add(m.group(1).strip())
            # 裸链
            for m in _URL_PAT.finditer(txt):
                urls.add(m.group("url").strip())
        except Exception:
            continue
    # 去掉结尾标点/转义残留
    cleaned = []
    for u in urls:
        u = u.strip().strip(").,;]")
        u = unquote(u)  # 有时会 url-encode
        cleaned.append(u)
    return sorted(set(cleaned))

def classify_links_via_llm(links: List[str]) -> List[Dict[str, str]]:
    """
    用 Qwen 粗分类：dataset / code / other
    —— 修复点：
       * 使用 _post_qwen_with_retry（长连接+重试+指数退避）
       * 读超时提升到 180s
       * 分批发送，降低超时概率
       * 失败时对该批次兜底为 other（不抛异常，避免卡住流水线）
    输出: [{"url":..., "label": "dataset|code|other", "why": "..."}]
    """
    if not links:
        return []

    results: List[Dict[str, str]] = []
    chunk_size = 4  # 可按需要调整

    for i in range(0, len(links), chunk_size):
        batch = links[i:i+chunk_size]
        # 和你原来的 prompt 尽量保持一致（只是换了调用方式）
        prompt = {
            "model": CONFIG.LLM_MODEL,
            "messages": [
                {
                    "role":"system",
                    "content":"你是信息抽取助手。给定URL列表，请判断其类型：dataset（数据集下载/主页）、code（项目官方代码仓库）、other（其他）。仅返回JSON。"
                },
                {"role":"user","content": json.dumps({"links": batch}, ensure_ascii=False)}
            ],
            "temperature": 0.1
        }

        try:
            data = _post_qwen_with_retry(prompt, timeout=(10, 180), tries=3)
            content = data["choices"][0]["message"]["content"]
            i0, j0 = content.find("{"), content.rfind("}") + 1
            batch_out: List[Dict[str, str]] = []
            if i0 >= 0 and j0 > i0:
                obj = json.loads(content[i0:j0])
                for it in obj.get("items", []):
                    u = (it.get("url","") or "").strip()
                    if not u:
                        continue
                    label = (it.get("label","other") or "other").lower()
                    why = it.get("why","")
                    batch_out.append({"url": u, "label": label, "why": why})

            # 如果模型没按预期返回，兜底给本批全部 other（可审计再重跑）
            if not batch_out:
                batch_out = [{"url": u, "label":"other", "why":"llm_empty_or_unparsable"} for u in batch]

            results.extend(batch_out)

        except Exception as e:
            logger.error("classify_links_via_llm error on batch %s-%s: %s", i, i+len(batch)-1, e)
            # 兜底：不要抛异常，给这一批标 other，流程不中断
            results.extend([{"url": u, "label":"other", "why": f"llm_timeout_or_error:{type(e).__name__}"} for u in batch])

    # 去重（同一 URL 以最后一次为准）
    dedup = {}
    for r in results:
        dedup[r["url"]] = r
    return list(dedup.values())

def _count_tokens(tokenizer, texts, sep_token):
    """
    估算拼接后的 token 数（不含 [CLS]/[SEP]），用于贴近 SentenceWindowDataset 的窗口估算逻辑。
    """
    if not texts:
        return 0
    # 中间 sep 作为一个 token 近似（不同模型略有差，但足够用于预算）
    joined = f" {sep_token} ".join([t for t in texts if t and t.strip()])
    return len(tokenizer.encode(joined, add_special_tokens=False))

def _build_single_window_around_pivot(
    tokenizer,
    sep_token,
    before_sents,              # list[str]
    after_sents,               # list[str]
    pivot_text,                # str, 用 "[URL_PIVOT] xxx" 这种短句占位
    max_length=512,
    min_sentences=3,
    context_size=2
):
    """
    以 URL 枢轴为中心，先确保前后 context_size 个句子，再在 token 预算内向两侧扩张，
    直到达到 min_sentences 或接近 max_length。
    """
    # 1) 原始序列：before + [pivot] + after
    before = [s for s in before_sents if s and s.strip()]
    after  = [s for s in after_sents  if s and s.strip()]
    sentences = before + [pivot_text] + after
    if not sentences:
        return []

    pivot_idx = len(before)  # 枢轴在 sentences 中的位置
    # 2) 先取枢轴附近的最小上下文
    start = max(0, pivot_idx - context_size)
    end   = min(len(sentences), pivot_idx + context_size + 1)

    # 3) 若还不够 min_sentences，则继续向两侧扩展
    def token_ok(slice_sents):
        # 预留 [CLS]/[SEP] 两个位置信息，粗略在  max_length-2 以内
        return _count_tokens(tokenizer, slice_sents, sep_token) <= (max_length - 2)

    cur_start, cur_end = start, end
    # 保证至少 min_sentences；但同时遵守 token 限制
    while (cur_end - cur_start) < min_sentences:
        # 尝试向左扩一格
        if cur_start > 0:
            if token_ok(sentences[cur_start-1:cur_end]):
                cur_start -= 1
            else:
                break
        # 尝试向右扩一格
        if (cur_end < len(sentences)) and ((cur_end - cur_start) < min_sentences):
            if token_ok(sentences[cur_start:cur_end+1]):
                cur_end += 1
            else:
                break
        if cur_start == 0 and cur_end == len(sentences):
            break  # 触达边界

    # 4) 在 token 预算允许下，尽可能继续向两边扩展，直到接近上限
    expanded = True
    while expanded:
        expanded = False
        # 优先左扩
        if cur_start > 0 and token_ok(sentences[cur_start-1:cur_end]):
            cur_start -= 1
            expanded = True
        # 再右扩
        if cur_end < len(sentences) and token_ok(sentences[cur_start:cur_end+1]):
            cur_end += 1
            expanded = True

    window_sents = sentences[cur_start:cur_end]
    return window_sents



# ==== 1 爬取的数据集 ====
def _utcnow():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def _arxiv_timefmt(dt_utc: datetime) -> str:
    # arXiv 需要 YYYYMMDDHHMM（UTC）
    return dt_utc.strftime("%Y%m%d%H%M")

def build_arxiv_range_query(cat: str, days: int, field: str) -> Dict[str, str]:
    now = _utcnow()
    start = now - timedelta(days=days)
    # 用空格拼接，交给 requests 负责编码
    qrange = f"{field}:[{_arxiv_timefmt(start)} TO {_arxiv_timefmt(now)}]"
    search_query = f"cat:{cat} AND {qrange}"
    return {
        "search_query": search_query,
        "sortBy": field,               # 'lastUpdatedDate' 或 'submittedDate'
        "sortOrder": "descending",     # 调试更直观；业务上随意
        "start": 0,
        "max_results": CONFIG.ARXIV_MAX_RESULTS,
    }

def _session():
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

def now_iso()->str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def ensure_dir(p:str):
    os.makedirs(p, exist_ok=True)

def today_str()->str:
    return datetime.now().strftime("%Y-%m-%d")

# ==== 统一“记账日期” ====
def day_str()->str:
    return CONFIG.TARGET_DAY or today_str()

def ledger_day_name() -> str:
    """
    目录/文件用的“记账日展示名”
    例：2025-10-15（7） —— 当 LAST_N_DAYS=7 时
       2025-10-15       —— 当 LAST_N_DAYS is None
    """
    base = day_str()  # 仍然由 TARGET_DAY 或 本机当天决定
    if getattr(CONFIG, "LAST_N_DAYS", None):
        return f"{base}（{CONFIG.LAST_N_DAYS}）"   # 全角括号
    return base

def paper_day_label(entry: dict) -> str:
    """
    若你希望“按论文本身的日期”组织目录（而非记账日），可以用这个。
    当前默认用 updated 的日期，缺失时回落到 day_str()。
    仍然会在有 LAST_N_DAYS 时附加括号。
    """
    base = (entry.get("updated", "") or entry.get("published", ""))[:10] or day_str()
    if getattr(CONFIG, "LAST_N_DAYS", None):
        return f"{base}（{CONFIG.LAST_N_DAYS}）"
    return base

def arxiv_id_from_pdf(pdf_url:str)->str:
    # https://arxiv.org/pdf/2509.01234v1.pdf
    m = re.search(r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5}(v\d+)?)\.pdf', pdf_url)
    return m.group(1) if m else ""

def pdf_link_from_id(arxiv_id:str)->str:
    # 支持 v 版本号
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def normalize_pdf_link(url:str)->str:
    base = url.split('#')[0].split('?')[0]
    # /abs/xxx -> /pdf/xxx.pdf
    m = re.search(r'arxiv\.org/(abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(v\d+)?)', base)
    if m:
        return f"https://arxiv.org/pdf/{m.group(2)}.pdf"
    if 'arxiv.org/html/' in base:
        tail = base.split('/html/')[1]
        return f"https://arxiv.org/pdf/{tail}.pdf"
    return base

# ==== 轻量状态存储（JSON 文件）====
class JsonState:
    def __init__(self, path:str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self._data = {"last_updated": {}, "seen_ids": {}}  # per category
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                logger.warning("State load failed, start fresh")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_last_updated(self, cat:str)->str:
        return self._data["last_updated"].get(cat, "")

    def set_last_updated(self, cat:str, ts:str):
        self._data["last_updated"][cat] = ts

    def seen(self, cat:str)->Set[str]:
        return set(self._data["seen_ids"].get(cat, []))

    def add_seen(self, cat:str, arxiv_id:str):
        s = self._data["seen_ids"].setdefault(cat, [])
        if arxiv_id not in s:
            s.append(arxiv_id)

# ==== arXiv 拉取（API，按提交时间降序）====
def fetch_arxiv_recent(cat: str, max_results: int = 10,start: int = 0) -> List[Dict]:
    """
    返回 [{id, title, abstract, updated, published, pdf_link}]
    支持：若配置了 LAST_N_DAYS，则按时间段过滤；否则保留原“recent”逻辑。
    """
    if CONFIG.LAST_N_DAYS:
        params = build_arxiv_range_query(cat, CONFIG.LAST_N_DAYS, CONFIG.ARXIV_DATE_FIELD)
        params["max_results"] = max_results
        params["start"] = start
        
    else:
        params = {
            "search_query": f"cat:{cat}",
            "sortBy": CONFIG.ARXIV_DATE_FIELD,  # 与 LAST_N_DAYS 分支保持一致
            "sortOrder": "descending",
            "start": start,  # 支持分页偏移
            "max_results": max_results,
        }

    s = _session()
    s.headers.update({"User-Agent": "DC-realtime-pipeline/1.0 (mailto:1135972876@qq.com)"})
    # 用 https
    r = s.get(CONFIG.ARXIV_API, params=params, timeout=40)
    r.raise_for_status()

    # 关键调试：看看实际发出的 URL（应与你浏览器那条等价，只是空格会显示成 +）
    logger.info("REQ URL: %s", r.url)

    # 统计 totalResults
    root_tmp = ET.fromstring(r.text)
    ns_total = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}
    total = root_tmp.findtext("opensearch:totalResults", namespaces=ns_total)
    logger.info("API ok: cat=%s, start=%s, max=%s, totalResults=%s",
                cat, params.get("start"), params.get("max_results"), total)

    
    
    # print(r.raise_for_status())

    root = ET.fromstring(r.text)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    def _ft(e, path, default=""):
        return (e.findtext(path, default=default, namespaces=ns) or "").strip()

    out = []
    for entry in root.findall("a:entry", ns):
        id_text = _ft(entry, "a:id")
        arxiv_id = id_text.split("/abs/")[-1].strip() if (id_text and "/abs/" in id_text) else (id_text.rsplit("/", 1)[-1].strip() if id_text else "")
        title = _ft(entry, "a:title")
        abstract = _ft(entry, "a:summary")
        updated = _ft(entry, "a:updated")
        published = _ft(entry, "a:published")
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        if not pdf_link:
            for link in entry.findall("a:link", ns):
                href = link.attrib.get("href", ""); typ = link.attrib.get("type", "")
                if typ == "application/pdf" or href.endswith(".pdf"):
                    pdf_link = href; break
        out.append({"id": arxiv_id, "title": title, "abstract": abstract,
                    "updated": updated, "published": published, "pdf_link": pdf_link})
    return out

import re, os, json, urllib.parse
from typing import List, Dict
from urllib.parse import unquote

_SENT_SPLIT = re.compile(r'(?<=[。！？.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    text = text.replace('\r', '').strip()
    # 合并中英文句号、问号、感叹号的正则
    pattern = re.compile(r'(?<=[。！？.!?])\s+')
    parts = pattern.split(text)
    if len(parts) <= 1:
        parts = [x for x in text.split('\n') if x.strip()]
    return [p.strip() for p in parts if p.strip()]


def parse_links_with_sentence_context(src_dir: str, window: int = 2) -> List[Dict]:
    """
    扫描 src_dir 下的 .tex/.bib/.txt/.md，提取每条链接，并给出“前后各 window 句”的上下文。
    返回元素示例：
    {
      "url": "https://example.org/data.zip",
      "file": "paper/main.tex",
      "line": 312,                 # 大致行号，便于定位
      "anchor": "Download data",   # \href{...}{这里}
      "section": "Dataset",        # 最近 \section 名称
      "ctx_before": ["句子-2", "句子-1"],
      "ctx_after":  ["句子+1", "句子+2"]
    }
    """
    href_pat = re.compile(r'\\href\{([^}]+)\}\{([^}]*)\}')
    url_pat  = re.compile(r'\\url\{([^}]+)\}')
    sec_pat  = re.compile(r'\\(?:sub)*section\*?\{([^}]*)\}')
    # 你现有的“遍历文本文件”的辅助（若叫别名请替换）
    def _iter_text_files(d):
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.lower().endswith(('.tex','.bib','.txt','.md')):
                    yield os.path.join(root, fn)

    out = []
    for fp in _iter_text_files(src_dir):
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
        except Exception:
            continue

        # 行号索引
        line_starts = [0]
        for m in re.finditer(r'\n', txt):
            line_starts.append(m.end())
        import bisect
        def line_no(pos):
            return bisect.bisect_right(line_starts, pos)

        # 章节索引
        sections = []
        for sm in sec_pat.finditer(txt):
            sections.append((sm.start(), sm.group(1).strip()))
        def nearest_section(pos):
            prev = [s for s in sections if s[0] <= pos]
            return prev[-1][1] if prev else ""

        # 预切句
        sents = _split_sentences(txt)
        # 建句子起止位置，便于匹配“某位置归属哪一句”
        spans = []
        acc = 0
        for s in sents:
            start = txt.find(s, acc)
            if start < 0:
                start = acc
            end = start + len(s)
            spans.append((start, end))
            acc = end

        def sent_index_at(pos: int) -> int:
            # 找到 pos 所在的句子 idx
            for i, (st, ed) in enumerate(spans):
                if st <= pos < ed:
                    return i
            # 边界情况
            if pos >= spans[-1][1]:
                return len(spans)-1
            return 0

        def grab_ctx(idx: int):
            b = [sents[i] for i in range(max(0, idx-2), idx)]
            a = [sents[i] for i in range(idx+1, min(len(sents), idx+3))]
            return b[-2:], a[:2]

        # 1) \href
        for m in href_pat.finditer(txt):
            url, anch = m.group(1).strip(), m.group(2).strip()
            pos = m.start()
            idx = sent_index_at(pos)
            before, after = grab_ctx(idx)
            out.append({
                "url": unquote(url).strip().strip(").]"),
                "file": os.path.relpath(fp, src_dir),
                "line": line_no(pos),
                "anchor": anch,
                "section": nearest_section(pos),
                "ctx_before": before,
                "ctx_after": after,
            })
        # 2) \url
        for m in url_pat.finditer(txt):
            url = m.group(1).strip()
            pos = m.start()
            idx = sent_index_at(pos)
            before, after = grab_ctx(idx)
            out.append({
                "url": unquote(url).strip().strip(").]"),
                "file": os.path.relpath(fp, src_dir),
                "line": line_no(pos),
                "anchor": "",
                "section": nearest_section(pos),
                "ctx_before": before,
                "ctx_after": after,
            })
        # 3) 裸链（沿用你原来的 URL 正则 _URL_PAT，如果已有就复用）
        _URL_PAT = re.compile(r'(?P<url>https?://[^\s{}\\]+)', re.I)
        for m in _URL_PAT.finditer(txt):
            url = m.group('url').strip()
            pos = m.start()
            idx = sent_index_at(pos)
            before, after = grab_ctx(idx)
            out.append({
                "url": unquote(url).strip().strip(").]"),
                "file": os.path.relpath(fp, src_dir),
                "line": line_no(pos),
                "anchor": "",
                "section": nearest_section(pos),
                "ctx_before": before,
                "ctx_after": after,
            })
    return out

def classify_link_inputs_via_llm(examples: List[Dict]) -> List[Dict]:
    """
    使用 Qwen 对“URL+上下文”进行分类（dataset|code|other）。
    输入：[{ "url":..., "input":... }，或包含 parse_links_with_sentence_context 的字段]
    输出：[{ "url":..., "label": "dataset|code|other", "why": "...", "desc": "..." }]
    """
    import json, re
    from typing import List, Dict, Tuple, Any

    chunk_size = 4
    if not examples:
        return []

    def _build_context(ex: Dict) -> str:
        """兼容不同来源的字段，优先使用 ex['input']，否则拼 section/anchor/上下文。"""
        txt = (ex.get("input") or "").strip()
        if not txt:
            parts = []
            if ex.get("anchor"):  parts.append(f"锚文本: {ex['anchor']}")
            if ex.get("section"): parts.append(f"章节: {ex['section']}")
            if ex.get("ctx_before"): parts.append("前文: " + " ".join(ex.get("ctx_before", [])))
            if ex.get("ctx_after"):  parts.append("后文: " + " ".join(ex.get("ctx_after", [])))
            txt = " | ".join([p for p in parts if p])
        # 压缩空白并限长，避免 prompt 过长
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt[:800]

    def _desc_fallback(ctx: str) -> str:
        """当 LLM 没给 desc 或解析失败时，基于上下文提炼一个≤120字的简述。"""
        ctx = (ctx or "").strip()
        if not ctx:
            return ""
        # 按句切两句；若没有句号类分隔，就截断 120
        parts = re.split(r'(?<=[。.!?])\s+', ctx)
        base = " ".join(parts[:2]).strip() or ctx[:120].strip()
        return (base[:120] + "…") if len(base) > 120 else base

    # 预处理成统一结构
    items = [{"url": (ex.get("url") or "").strip(), "context": _build_context(ex)} for ex in examples]

    results: List[Dict[str, str]] = []
    allowed = {"dataset", "code", "other"}

    # ------- 解析辅助：把各种 content 形态统一成 JSON 对象/数组 -------
    def _strip_code_fence(s: str) -> str:
        # 去掉 ```json ... ``` 或 ``` ... ```
        return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.IGNORECASE)

    def _join_content_fragments(content_any) -> str:
        """兼容 message.content 为 list 分片的情况，抽取 text 段拼接。"""
        if isinstance(content_any, str):
            return content_any
        if isinstance(content_any, list):
            parts = []
            for p in content_any:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("output_text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return str(content_any)

    def _first_top_level_span(s: str, open_ch: str, close_ch: str) -> str:
        """寻找首个顶层 open_ch..close_ch 片段（计数法，忽略字符串内的括号）。"""
        start = s.find(open_ch)
        if start == -1:
            return ""
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        return ""

    def _maybe_fix_over_escaped_quotes(s: str) -> str:
        """针对 \\\\\" 这类过度转义做一次谨慎修复，仅作为最后尝试。"""
        t = _strip_code_fence(s).replace('\ufeff', '').strip()
        if t.count('\\\\\\"') >= 2 and t.count('\\"') == 0:
            t = t.replace('\\\\\\"', '\\"')
        return t

    def _as_items(obj: Any) -> List[Dict]:
        """把 obj 统一转成 items 列表。"""
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("items"), list):
                return obj["items"]
            if isinstance(obj.get("data"), list):
                return obj["data"]
            if {"url", "label"} <= set(obj.keys()):
                return [obj]
        return []

    def _parse_items_from_content(content_any) -> List[Dict]:
        """
        鲁棒解析：整体 -> 去围栏 -> 顶层数组 -> 顶层对象 -> 过度转义修复后重试。
        """
        text = _join_content_fragments(content_any).replace('\ufeff', '').strip()

        # 1) 直接整体解析
        try:
            obj = json.loads(text)
            return _as_items(obj)
        except Exception:
            pass

        # 2) 去掉代码围栏再解析
        t2 = _strip_code_fence(text).strip()
        try:
            obj = json.loads(t2)
            return _as_items(obj)
        except Exception:
            pass

        # 3) 优先尝试首个顶层数组片段
        frag = _first_top_level_span(t2, '[', ']')
        if frag:
            try:
                obj = json.loads(frag)
                return _as_items(obj)
            except Exception:
                pass

        # 4) 再尝试顶层对象片段（以兼容 {"items":[...]} / {"data":[...]}）
        frag = _first_top_level_span(t2, '{', '}')
        if frag:
            try:
                obj = json.loads(frag)
                items_ = _as_items(obj)
                if items_:
                    return items_
            except Exception:
                pass

        # 5) 尝试“过度转义”修复后，重复 1-4
        t3 = _maybe_fix_over_escaped_quotes(text)
        if t3 != text:
            for cand in (t3, _strip_code_fence(t3).strip()):
                try:
                    obj = json.loads(cand)
                    return _as_items(obj)
                except Exception:
                    frag = _first_top_level_span(cand, '[', ']')
                    if frag:
                        try:
                            obj = json.loads(frag)
                            return _as_items(obj)
                        except Exception:
                            pass
                    frag = _first_top_level_span(cand, '{', '}')
                    if frag:
                        try:
                            obj = json.loads(frag)
                            items_ = _as_items(obj)
                            if items_:
                                return items_
                        except Exception:
                            pass
        # 全部失败
        return []

    # -------- 主流程：并发节流下，按批处理 ----------
    with _LLM_SEM:
        for i in range(0, len(items), chunk_size):
            batch = items[i:i + chunk_size]

            # 组织“只返回 JSON”的紧凑指令
            user_obj = {
                "task": "classify_urls",
                "labels": ["dataset", "code", "other"],
                "definition": {
                    "dataset": "论文作者首次提出/专门为本研究构建或发布的数据集主页/下载页/DOI/平台地址",
                    "code":    "该论文项目的官方代码实现或仓库（如 GitHub、HF 等）",
                    "other":   "不属于上述两类或与论文无直接对应关系的链接"
                },
                "rules": [
                    "逐项判断并按输入顺序输出；只输出 JSON 数组或 {items:[...]}",
                    "每项返回: url, label ∈ {dataset,code,other}, why(≤16字), desc(≤80字)",
                    "desc：基于给定上下文，1–2 句说明该链接对应的数据集/用途；不要复述整段",
                    "若同一链接既有数据又有代码，只挑最合适的一类",
                    "若无法判断，标记为 other"
                ],
                "items": batch  # [{url, context}]
            }

            payload = {
                "model": CONFIG.LLM_MODEL,
                "messages": [
                    {"role": "system",
                     "content": "你是信息抽取助手。严格按要求‘仅返回 JSON’，不要输出任何额外文本。"},
                    {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}
                ],
                "temperature": 0.1,
            }

            try:
                data = _post_qwen_with_retry(payload, timeout=(10, 180), tries=3)
                content = data["choices"][0]["message"]["content"]

                # ——稳健 JSON 提取——
                parsed_items = _parse_items_from_content(content)

                # 规范化与对齐输入顺序；若缺失则兜底
                if len(parsed_items) != len(batch):
                    logger.warning("classify_link_inputs_via_llm: LLM 返回数量不匹配，使用兜底启发式。")

                def _heuristic(ctx: str) -> Tuple[str, str]:
                    low = (ctx or "").lower()
                    if any(k in low for k in [
                        " dataset", "data set", "corpus", "benchmark", "annotations",
                        "download data", "数据集", "标注", "下载数据", "doi", "zenodo", "figshare", "mendeley",
                        "huggingface.co/datasets", "osf.io", "openml", "dataverse"
                    ]):
                        return "dataset", "关键词触发"
                    if any(k in low for k in [
                        " github", "source code", "implementation", "repo", "repository",
                        "代码", "实现", "huggingface.co", "gitlab"
                    ]):
                        return "code", "关键词触发"
                    return "other", "无明显线索"

                # 映射到每个输入；若解析失败/缺失，对该项做兜底判断
                for idx, b in enumerate(batch):
                    if idx < len(parsed_items):
                        it = parsed_items[idx] or {}
                        url  = (it.get("url") or b["url"] or "").strip()
                        lab  = (it.get("label") or "other").lower().strip()
                        why  = (it.get("why") or "").strip()
                        desc = (it.get("desc") or "").strip()

                        # 归一化 label
                        if "data" in lab:   lab = "dataset"
                        if any(k in lab for k in ("code", "repo", "github")): lab = "code"
                        if lab not in allowed:
                            lab, why = _heuristic(b["context"])

                        if not why:
                            why = "上下文判别"
                        if not desc:
                            desc = _desc_fallback(b["context"])

                        results.append({"url": url or b["url"], "label": lab, "why": why[:24], "desc": desc})
                    else:
                        # 兜底启发式
                        lab, why = _heuristic(b["context"])
                        results.append({
                            "url": b["url"],
                            "label": lab,
                            "why": why,
                            "desc": _desc_fallback(b["context"])
                        })

            except Exception as e:
                logger.error("classify_link_inputs_via_llm error on batch %d-%d: %s",
                             i, i + len(batch) - 1, e)
                # 批量兜底（保序并回填 URL）
                for b in batch:
                    results.append({
                        "url": b["url"],
                        "label": "other",
                        "why": f"llm_timeout_or_error:{type(e).__name__}"[:24],
                        "desc": _desc_fallback(b["context"])
                    })

    # 最终一致性校验（极端情况下长度不等）
    if len(results) != len(items):
        logger.warning("classify_link_inputs_via_llm: 结果数与输入数不一致，进行修补。")
        # 截断或补齐
        if len(results) > len(items):
            results = results[:len(items)]
        else:
            for k in range(len(results), len(items)):
                ck = items[k]
                results.append({
                    "url": ck["url"],
                    "label": "other",
                    "why": "mismatch_fallback",
                    "desc": _desc_fallback(ck.get("context", ""))
                })

    return results


# ==== 2 BERT 分类（title+abstract 是否“可能自建数据集”）====
import torch, torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

class BertGate:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_TOKENIZER)
        self.model = BertForSequenceClassification.from_pretrained(CONFIG.BERT_CKPT)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def score(self, title:str, abstract:str)->float:
        text = (title or "") + " " + (abstract or "")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prob = F.softmax(logits, dim=-1)[0,1].item()
        return float(prob)

# ==== 3 Data_Description 提取部分====
#改进了一下，分section保存
def grobid_extract_to_txt(pdf_path: str, txt_path: str) -> bool:
    open(txt_path, "w", encoding="utf-8").close()
    files = {"input": open(pdf_path, "rb")}
    try:
        r = requests.post(CONFIG.GROBID_URL, files=files, timeout=60)
        if r.status_code != 200:
            logger.error("GROBID failed %s", r.status_code)
            return False
        
        # 用 XML 解析而不是去掉标签
        root = ET.fromstring(r.text)
        sections = []
        for div in root.findall(".//{*}div"):
            head = div.find(".//{*}head")
            title = head.text.strip() if head is not None else ""
            # 拼所有段落
            paragraphs = [p.text.strip() for p in div.findall(".//{*}p") if p.text]
            if paragraphs:
                text = " ".join(paragraphs)
                sections.append((title, text))

        # 如果没解析出结构，就退回全文本
        if not sections:
            plain = re.sub(r"<[^>]+>", " ", r.text)
            plain = re.sub(r"\s+", " ", plain).strip()
            sections = [("", plain)]

        title = os.path.splitext(os.path.basename(pdf_path))[0]
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"[TITLE]{title}[/TITLE]\n")
            for sec_title, sec_text in sections:
                f.write(f"[SECTION]{sec_title}[/SECTION]\n{sec_text}\n")
        return True
    except Exception as e:
        logger.exception("grobid_extract_to_txt error: %s", e)
        return False
    finally:
        try:
            files["input"].close()
        except Exception:
            pass 

def txt_to_json(txt_path: str, json_path: str) -> bool:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        title_match = re.search(r"\[TITLE\](.*?)\[/TITLE\]", content)
        paper_name = title_match.group(1).strip() if title_match else os.path.basename(txt_path)

        # 按 section 切分
        sections = re.split(r"\[SECTION\](.*?)\[/SECTION\]", content)
        # sections = ['', 'section1 title', 'section1 text', 'section2 title', 'section2 text', ...]
        data_items = []
        for i in range(1, len(sections), 2):
            sec_title = sections[i].strip()
            sec_body = sections[i+1].strip()
            sents = _split_sentences(sec_body)
            data_items.append({
                "paper_name": paper_name,
                "section": sec_title,
                "paragraph_id": i//2,
                "full_text": sec_body,
                "sentences": [
                    {"sentence_id": j, "text": s, "label": 0} for j, s in enumerate(sents)
                ]
            })

        data = {"data": data_items}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.exception("txt_to_json error: %s", e)
        return False



# ==== LLM：句子贴标签（label==1 代表“数据集描述”）====
def llm_client():
    api_key = os.environ.get(CONFIG.DASHSCOPE_API_KEY_ENV)
    assert api_key, "Please export DASHSCOPE_API_KEY_ENV"
    s = _session()
    s.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    })
    return s

# ========= 提示词构建（整篇 ≤100句） =========
def format_paper_for_prompt(self, paper: Dict, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
    title = paper.get('paper_name', '')
    sentences = paper.get('sentences', [])
    if end_idx is None:
        end_idx = len(sentences)

    total_sentences = end_idx - start_idx
    header = f"""请你作为“论文数据集描述识别助手”，逐句判断给定论文文本中每个句子是否描述了“该研究特定构建或使用的数据集”。

论文标题：{title}

【严格输出要求】
1) 必须为本次范围内的每个句子逐条标注，不能遗漏、不能重复。
2) 仅按如下格式输出，每行一条： 句子X,标记为Y。解释：Z
   - X：句子编号（从 {start_idx+1} 到 {end_idx}）
   - Y ∈ {{0,1}}；1 表示“数据集描述”，0 表示“非数据集描述”
   - Z：简短原因（不超过20字），不要复述原句全文
3) 输出行数必须等于 {total_sentences}，且编号与句子一一对应。
4) 只输出标注行，不要夹杂额外说明、列表符号或JSON。

【判定为1（数据集描述）的标准，满足其一即可】
- 描述本研究“新建/使用的特定数据集”的：构建/采集/来源/预处理/规模/组成/标注方式/质量控制/发布与获取方式（链接或地址）
- 直接或间接描述本研究所用到的特定数据的细节

【应判为0的常见情形】
- 模型方法、训练细节（超参/优化器/损失函数/结构）而不涉及数据来源或构建
- 指标与实验结果/消融/可视化等
- 相关工作综述或通用背景介绍
- 只出现公共基准名称但未描述本研究“特定数据”的构建/使用细节

下面开始逐句标注（仅输出“句子X,标记为Y。解释：Z”）：\n"""
    lines = []
    for i in range(start_idx, end_idx):
        lines.append(f"句子{i+1}: {sentences[i]['text']}")
    return header + "\n".join(lines)


# ========= 分块提示词（>100句时使用） =========
def build_chunk_prompt(paper_name: str, chunk_sentences: List[Dict], start_idx: int) -> str:
    end_idx = start_idx + len(chunk_sentences)
    header = f"""你是“论文数据集描述识别助手”。请仅标注编号从 {start_idx+1} 到 {end_idx} 的这些句子。
论文标题：{paper_name}

【必须遵守的输出格式（每行一条）】
句子X,标记为Y。解释：Z
- X ∈ [{start_idx+1}, {end_idx}]
- Y ∈ {{0,1}}（1=数据集描述；0=非）
- Z 简短，不超过20字；不要粘贴原句；不要输出其它多余内容
- 输出行数必须等于 {len(chunk_sentences)}，且包含本块的全部编号

【判定为1（数据集描述）示例】
- “我们收集了X来源的数据并人工标注……”
- “本研究新建了包含N张图像/样本的Y数据集……”
- “数据可从链接/仓库下载（给出地址/平台）……”

【判定为0的反例】
- “我们使用Transformer + AdamW……”
- “实验在COCO/Imagenet上取得……（若未涉及本研究的特定数据构建/使用细节）”
- “相关工作回顾/评价指标/可视化结论……”

请按顺序仅输出标注行：\n"""
    lines = []
    for i, obj in enumerate(chunk_sentences, start=start_idx+1):
        lines.append(f"句子{i}: {obj['text']}")
    return header + "\n".join(lines)

# ========= 使用分块提示词的 LLM 贴标签 =========
def label_sentences_via_llm(json_path: str) -> bool:
    """
    加入并发限流（全函数被信号量包裹，最大并发由 CONFIG.MAX_LLM_WORKERS 控制）
    """
    with _LLM_SEM:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sents = data["data"][0]["sentences"]
            paper_name = data["data"][0].get("paper_name", "")

            chunk = 100
            pat = re.compile(r"句子\s*(\d+)\s*[,，．.、]?\s*标记为\s*([01])", re.IGNORECASE)

            def call_once(prompt_text: str, timeout_sec: int = 90) -> str:
                payload = {
                    "model": CONFIG.LLM_MODEL,
                    "messages": [
                        {"role": "system",
                         "content": "你是论文数据集描述识别助手。请严格按要求输出：仅逐行输出“句子X,标记为Y。解释：Z”。禁止输出其它内容。"},
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": CONFIG.LLM_TEMPERATURE,
                }
                # r = client.post(CONFIG.LLM_BASE_URL + "/chat/completions", json=payload, timeout=timeout_sec)
                # r.raise_for_status()
                # return r.json()["choices"][0]["message"]["content"]
                data = _post_qwen_with_retry(payload, timeout=(10, max(120, timeout_sec)), tries=3)
                return data["choices"][0]["message"]["content"]

            def parse_and_apply(content: str, start_idx: int, end_idx: int) -> int:
                count = 0
                for line in content.splitlines():
                    m = pat.search(line)
                    if not m:
                        continue
                    gidx = int(m.group(1)) - 1  # 全局0-based
                    y = int(m.group(2))
                    if start_idx <= gidx < end_idx and y in (0, 1):
                        sents[gidx]["label"] = y
                        count += 1
                return count

            for b in range(0, len(sents), chunk):
                part = sents[b:b+chunk]
                prompt_text = build_chunk_prompt(paper_name, part, start_idx=b)

                content = call_once(prompt_text, timeout_sec=90)
                ok = parse_and_apply(content, b, b + len(part))

                if ok != len(part):
                    fix_prompt = "上一次输出的行数或编号不匹配。请仅按严格格式重新输出本块的全部标注：\n" + prompt_text
                    content2 = call_once(fix_prompt, timeout_sec=90)
                    ok2 = parse_and_apply(content2, b, b + len(part))
                    ok = max(ok, ok2)

                if ok != len(part):
                    logger.warning("LLM 标注数量仍不匹配：%s / %s（块起始 %s）", ok, len(part), b)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.exception("label_sentences_via_llm error: %s", e)
            return False


# ==== 4  LLM：从全文抽“数据集链接” ====
def extract_dataset_links_via_llm(txt_path:str)->List[str]:
    with _LLM_SEM:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
            client = llm_client()
            user_prompt = f'''请从以下论文文本中提取该论文为了该研究而专门制作的数据集的URL，包括：

    论文全文：
    {paper_text}

    请按以下步骤分析论文：

    第一步：识别论文信息
    1. 确定输入论文的标题是什么
    2. 确定该论文的第一作者(first author)是谁
    3. 该论文发表于哪个时间

    第二步：理解研究内容
    1. 这篇论文的主要研究工作是什么
    2. 作者为了完成这项研究，创建了什么原创数据集
    3. 论文作者是否开源了项目代码？如果开源，在论文中的哪里提到了代码仓库链接

    第三步：提取关键链接
    请仅提取以下两类链接：

    1. 数据集链接 - 必须同时满足：
    - 该数据集是论文作者首次在本论文中提出的
    - 是为本研究专门创建的数据集
    - 论文中明确说明这是新的(new)或首次发布的(introduce/propose/present)数据集
    - 提供了明确的下载链接

    2. 官方代码仓库链接 - 必须满足：
    - 是论文作者开源的本项目的官方代码实现
    - 通常在论文中用"our code"、"source code"、"implementation"、"github"等词语引入
    - 位于论文正文、脚注或项目链接章节

    重要提示：
    - 请确保每个链接只返回一次，不要在dataset和code类别中包含重复相同的链接，请对最终的结果进行检查
    - 如果一个链接既包含数据集又包含代码，请只将其归类到一个最合适的类别中
    - 请仔细检查链接的有效性，确保它们是完整的URL

    请以JSON格式返回找到的链接:
    {{
        "urls": {{
            "dataset": ["数据集链接1", "数据集链接2", ...],
            "code": ["代码仓库链接1", "代码仓库链接2", ...]
        }}
    }}

    并说明为什么你认为返回的链接确实是输入论文的第一作者为该研究专门创建的数据集链接或者开源的代码链接。'''
            payload = {
                "model": CONFIG.LLM_MODEL,
                "messages":[
                    {"role":"system","content":"只返回 JSON；数据集必须为论文作者首次提出且提供下载链接。"},
                    {"role":"user","content": user_prompt}
                ],
                "temperature": 0.1
            }
            data = _post_qwen_with_retry(payload, timeout=(10, 100), tries=3)
            content = data["choices"][0]["message"]["content"]
            
            start, end = content.find("{"), content.rfind("}")+1
            urls = []
            if start>=0 and end>start:
                obj = json.loads(content[start:end])
                ds = obj.get("urls", {}).get("dataset", []) or []
                code = obj.get("urls", {}).get("code", []) or []
                urls = list({*ds, *code})
            keep_hosts = ("kaggle.com","zenodo.org","figshare.com","data.mendeley.com","huggingface.co","github.com")
            urls = [u for u in urls if any(h in u.lower() for h in keep_hosts)]
            return urls
        except Exception as e:
            logger.exception("extract_dataset_links_via_llm error: %s", e)
            return []

# ==== 从打标 JSON 汇总“数据集描述” ====
def collect_dataset_description(json_path:str, max_chars:int=2000)->str:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sents = []
        for s in data["data"][0]["sentences"]:
            if int(s.get("label",0)) == 1 and s.get("text"):
                sents.append(s["text"].strip())
        desc = " ".join(sents).strip()
        return (desc[:max_chars].rstrip() + " ...") if len(desc)>max_chars else desc
    except Exception:
        return ""

# ==== 写入“当日四列 JSON”，幂等去重 ====
def append_final_records(records:List[Dict]):
    # day = day_str()  # 按目标日记账
    day = ledger_day_name()
    
    out_dir = CONFIG.OUT_FINAL
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{day}.json")

    with _FINAL_WRITE_LOCK:  # 并发写保护
        existing = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        merged = existing + records
        seen, dedup = set(), []
        for r in merged:
            key = (r.get("Paper Link",""), r.get("Dataset Link",""))
            if key in seen:
                continue
            seen.add(key); dedup.append(r)
        #z这里有保存文件的逻辑
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dedup, f, ensure_ascii=False, indent=2)
    logger.info("Final records written: %s (+%d)", out_path, len(records))

# ==== PDF 下载 ====
def download_pdf(pdf_url: str, save_path: str, max_retries: int = 5) -> bool:
    try:
        pdf_url = normalize_pdf_link(pdf_url)

        # 构建带重试机制的 Session
        s = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1,  # 每次失败后等待1s,2s,4s...
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        s.mount("http://", HTTPAdapter(max_retries=retries))
        s.mount("https://", HTTPAdapter(max_retries=retries))

        # 下载尝试循环
        for attempt in range(1, max_retries + 1):
            try:
                r = s.get(pdf_url, stream=True, timeout=60)
                r.raise_for_status()
                ensure_dir(os.path.dirname(save_path))

                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"PDF downloaded successfully: {pdf_url}")
                return True  # 成功后返回

            except (ChunkedEncodingError, ConnectionError, ReadTimeout, SSLError) as e:
                logger.warning(
                    f"Download attempt {attempt}/{max_retries} failed: {type(e).__name__} - {e}"
                )
                # 等待后重试
                time.sleep(2 ** attempt)

        # 全部尝试失败
        logger.error(f"PDF download failed after {max_retries} attempts: {pdf_url}")
        return False

    except Exception as e:
        logger.exception("PDF download failed: %s", e)
        return False


# ==== 单篇论文重处理（不再做 BERT 阈值判断，已在主线程预筛） ====
def process_one_paper(cat:str, entry:Dict) -> bool:
    """
    entry: {id,title,abstract,updated,published,pdf_link}
    返回 True 表示整篇处理成功（仅成功才加入 seen）
    """
    arxiv_id = entry["id"]
    # 路径（按目标日记账）
    day = day_str()
    pdf_dir = os.path.join(CONFIG.OUT_PDF, day, cat)
    txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
    json_path = os.path.join(pdf_dir, f"{arxiv_id}.json")
    pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
    paper_link = pdf_link_from_id(arxiv_id)
    timestamp = now_iso()

    # 1) PDF
    if not os.path.exists(pdf_path):
        ok = download_pdf(paper_link, pdf_path)
        if not ok:
            return False
        time.sleep(1.0 + random.random())  # 轻微间隔

    # 2) GROBID → TXT
    if not os.path.exists(txt_path):
        ok = grobid_extract_to_txt(pdf_path, txt_path)
        if not ok:
            return False

    # 3) TXT → JSON（句子，默认 label=0）
    if not os.path.exists(json_path):
        if not txt_to_json(txt_path, json_path):
            return False

    # 4) LLM 贴标签（受并发限流）
    if not label_sentences_via_llm(json_path):
        return False

    # 5) LLM 抽数据集链接（受并发限流）
    links = extract_dataset_links_via_llm(txt_path)

    # 6) 汇总数据集描述
    dataset_desc = collect_dataset_description(json_path)

    # 7) 写最终四列（若无链接也写一条空链接）
    final_links = sorted(set([l for l in links if l])) or [""]
    records = [{
        "Paper Link": paper_link,
        "Dataset Link": lk,
        "Dataset Description": dataset_desc,
        "Timestamp": timestamp
    } for lk in final_links]
    append_final_records(records)
    return True

# ==== 主循环（并发） ====
def poll_and_process():
    # 初始化
    ensure_dir(CONFIG.STATE_DIR)
    state = JsonState(os.path.join(CONFIG.STATE_DIR, "arxiv_state.json"))
    ensure_dir(CONFIG.OUT_PAPERS)

    bert = BertGate()

    while True:
        start_loop = time.time()
        for cat in CONFIG.CATEGORIES:
            try:
                time.sleep(random.uniform(*CONFIG.JITTER_SEC))
                logger.info("Polling %s ...", cat)

                # 拉更多，避免当天条目被挤出
                all_entries = []
                start = 0
                while True:
                    page = fetch_arxiv_recent(cat, max_results=CONFIG.ARXIV_MAX_RESULTS, start=start)
                    if not page:
                        break
                    all_entries.extend(page)
                    if len(page) < CONFIG.ARXIV_MAX_RESULTS:
                        break
                    start += CONFIG.ARXIV_MAX_RESULTS
                entries = all_entries
                
                logger.info("Fetched %s entries in %s (LAST_N_DAYS=%s)", len(entries), cat, CONFIG.LAST_N_DAYS)
                # ========若指定了目标日，只保留该日发布的论文================
                if CONFIG.TARGET_DAY and not CONFIG.LAST_N_DAYS:
                    entries = [e for e in entries if e.get("published", "")[:10] == day_str()]
                # 当天元数据落盘（覆盖写，哪怕 entries 为空也写）
                meta_dir = os.path.join(CONFIG.OUT_PAPERS, day_str())
                ensure_dir(meta_dir)
                out_meta = os.path.join(meta_dir, f"{cat}.json")
                with open(out_meta, "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)
                logger.info("Wrote meta: %s (entries=%s)", out_meta, len(entries))
                
                # 若指定了目标日，只保留该日发布的论文（按 arXiv <published> 的日期）
                if CONFIG.TARGET_DAY and not CONFIG.LAST_N_DAYS:
                    entries = [e for e in entries if e.get("published", "")[:10] == day_str()]

                # 当天元数据落盘（覆盖写）
                meta_dir = os.path.join(CONFIG.OUT_PAPERS, day_str())
                ensure_dir(meta_dir)

                #============================

                # seen 逻辑：跑“指定日期”时可选择忽略历史去重
                seen = set() if (CONFIG.TARGET_DAY and CONFIG.IGNORE_SEEN_WHEN_TARGETED) else state.seen(cat)
                candidates = [e for e in entries if e["id"] not in seen]
                logger.info("Candidates after seen-filter: %s (seen=%s)", len(candidates), len(seen))
                
                
                
                # ===== 主线程先用 BERT 预筛 =====
                prefiltered = []
                for e in candidates:
                    prob = bert.score(e["title"], e["abstract"])
                    if prob >= CONFIG.BERT_THRESHOLD:
                        prefiltered.append(e)
                    else:
                        logger.info("[%s] %s filtered by BERT (%.2f)", cat, e["id"], prob)

                if not prefiltered:
                    logger.info("No entries to process for %s", cat)
                    continue

                # ===== 并发处理论文 =====
                futures = []
                with ThreadPoolExecutor(max_workers=CONFIG.MAX_PAPER_WORKERS) as ex:
                    for e in prefiltered:
                        futures.append(ex.submit(process_one_paper, cat, e))
                    for fut, e in zip(as_completed(futures), prefiltered):
                        ok = fut.result()
                        if ok:
                            state.add_seen(cat, e["id"])           # 仅成功后才记 seen
                            state.set_last_updated(cat, e["updated"])
                            state.save()

            except Exception as e:
                logger.exception("Poll/process error on %s: %s", cat, e)

        # 计算下一轮时间
        elapsed = time.time() - start_loop
        sleep_sec = max(0, CONFIG.POLL_MINUTES*60 - elapsed)
        logger.info("Cycle done, sleep %.1fs", sleep_sec)
        time.sleep(sleep_sec)

if __name__ == "__main__":
    """
    运行前请准备：
    1) export DASHSCOPE_API_KEY=xxxx
    2) GROBID 服务在本机 8071 端口
    3) BERT 模型放在 CONFIG 里的路径
    """
    poll_and_process()
