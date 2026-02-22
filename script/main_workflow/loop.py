# realtime_pipeline.py
# -*- coding: utf-8 -*-

import os, re, json, time, math, asyncio, logging, random, threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== 3) 路径与配置，统一到 kzlab 下面 ====
class CONFIG:
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
    ARXIV_API = "http://export.arxiv.org/F"

    # 轮询周期（分钟）
    POLL_MINUTES = 30

    # ======= 并发控制 =======
    # 同时并行处理的“论文数”（下载/GROBID/贴标签在每个任务内顺序执行）
    MAX_PAPER_WORKERS = 4
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

    # LLM（阿里灵积 DashScope 兼容 OpenAI）
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL = "qwen-plus"
    LLM_TEMPERATURE = 0.15
    DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"  # 请 export 到环境变量

    # 随机抖动（防止各类目同时打 API）
    JITTER_SEC = (0.0, 2.0)

    # ==== 目标日期与去重策略 ====
    TARGET_DAY = "2025-09-21"       # 想按系统当天请设为 None；或填 "YYYY-MM-DD"
    IGNORE_SEEN_WHEN_TARGETED = True  # 指定 TARGET_DAY 时是否忽略历史去重
    ARXIV_MAX_RESULTS = 200           # 每类目拉取更充足，避免当天条目被挤出前N

# ==== 日志 ====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("realtime")

# ==== 并发相关全局对象 ====
_LLM_SEM = threading.Semaphore(CONFIG.MAX_LLM_WORKERS)  # LLM 并发限流
_FINAL_WRITE_LOCK = threading.Lock()                    # 写 final json 的互斥锁

# ==== 工具 ====
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
def fetch_arxiv_recent(cat: str, max_results: int = 10) -> List[Dict]:
    """
    返回 [{id, title, abstract, updated, published, pdf_link}]
    """
    params = {
        "search_query": f"cat:{cat}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }
    s = _session()
    s.headers.update({"User-Agent": "DC-realtime-pipeline/1.0 (mailto:you@example.com)"})
    r = s.get(CONFIG.ARXIV_API, params=params, timeout=20)
    r.raise_for_status()

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        raise

    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def _ft(e, path, default=""):
        return (e.findtext(path, default=default, namespaces=ns) or "").strip()

    out = []
    for entry in root.findall("a:entry", ns):
        id_text = _ft(entry, "a:id")
        arxiv_id = ""
        if id_text:
            if "/abs/" in id_text:
                arxiv_id = id_text.split("/abs/")[-1].strip()
            else:
                arxiv_id = id_text.rsplit("/", 1)[-1].strip()

        title = _ft(entry, "a:title")
        abstract = _ft(entry, "a:summary")   # ✅ arXiv 用 <summary>
        updated = _ft(entry, "a:updated")
        published = _ft(entry, "a:published")

        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        if not pdf_link:
            for link in entry.findall("a:link", ns):
                href = link.attrib.get("href", "")
                typ = link.attrib.get("type", "")
                if typ == "application/pdf" or href.endswith(".pdf"):
                    pdf_link = href
                    break

        out.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "updated": updated,
            "published": published,
            "pdf_link": pdf_link,
        })
    return out



# ==== BERT 分类（title+abstract 是否“可能自建数据集”）====
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

# ==== GROBID 解析 ====
def grobid_extract_to_txt(pdf_path:str, txt_path:str)->bool:
    open(txt_path, "w", encoding="utf-8").close()
    files = {"input": open(pdf_path, "rb")}
    try:
        r = requests.post(CONFIG.GROBID_URL, files=files, timeout=60)
        if r.status_code != 200:
            logger.error("GROBID failed %s", r.status_code)
            return False
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        title = os.path.splitext(os.path.basename(pdf_path))[0]
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"[TITLE]{title}[/TITLE]\n{text}\n")
        return True
    except Exception as e:
        logger.exception("GROBID error: %s", e)
        return False
    finally:
        try:
            files["input"].close()
        except Exception:
            pass

# ==== 文本预处理：切句 → JSON（与您之前格式兼容）====
def split_sentences(text:str)->List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"\[])|$", text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def txt_to_json(txt_path:str, json_path:str)->bool:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"\[TITLE\](.*?)\[/TITLE\]", content)
        paper_name = m.group(1).strip() if m else os.path.basename(txt_path)
        body = re.sub(r"\[TITLE\].*?\[/TITLE\]", "", content, flags=re.S).strip()
        sents = split_sentences(body)
        data = {
            "data":[{
                "paper_name": paper_name,
                "section": "",
                "paragraph_id": 0,
                "full_text": body,
                "sentences":[{"sentence_id":i,"text":s,"label":0} for i,s in enumerate(sents)]
            }]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.exception("txt_to_json error: %s", e)
        return False

# ==== LLM：句子贴标签（label==1 代表“数据集描述”）====
def llm_client():
    api_key = os.environ.get(CONFIG.DASHSCOPE_API_KEY_ENV)
    assert api_key, "Please export DASHSCOPE_API_KEY"
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

            client = llm_client()
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
                r = client.post(CONFIG.LLM_BASE_URL + "/chat/completions", json=payload, timeout=timeout_sec)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]

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


# ==== LLM：从全文抽“数据集链接” ====
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
            r = client.post(CONFIG.LLM_BASE_URL + "/chat/completions", json=payload, timeout=90)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
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
    day = day_str()  # 按目标日记账
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
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dedup, f, ensure_ascii=False, indent=2)
    logger.info("Final records written: %s (+%d)", out_path, len(records))

# ==== PDF 下载 ====
def download_pdf(pdf_url:str, save_path:str)->bool:
    try:
        pdf_url = normalize_pdf_link(pdf_url)
        s = _session()
        r = s.get(pdf_url, stream=True, timeout=30)
        r.raise_for_status()
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
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
                entries = fetch_arxiv_recent(cat, max_results=CONFIG.ARXIV_MAX_RESULTS)

                # 若指定了目标日，只保留该日发布的论文（按 arXiv <published> 的日期）
                if CONFIG.TARGET_DAY:
                    entries = [e for e in entries if e.get("published", "")[:10] == day_str()]

                # 当天元数据落盘（覆盖写）
                meta_dir = os.path.join(CONFIG.OUT_PAPERS, day_str())
                ensure_dir(meta_dir)
                with open(os.path.join(meta_dir, f"{cat}.json"), "w", encoding="utf-8") as f:
                    json.dump(entries, f, ensure_ascii=False, indent=2)

                # seen 逻辑：跑“指定日期”时可选择忽略历史去重
                seen = set() if (CONFIG.TARGET_DAY and CONFIG.IGNORE_SEEN_WHEN_TARGETED) else state.seen(cat)
                candidates = [e for e in entries if e["id"] not in seen]

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
