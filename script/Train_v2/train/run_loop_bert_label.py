# -*- coding: utf-8 -*-
"""
合并版：将 `run_loop_bert_label.py` 与 `utils/utils_bertlabel_latex_extract.py` 的核心逻辑合并为单文件，
保留原有流程与函数命名，便于平滑替换。

依赖：仍依赖 `utils.utils_loop` 中的工具/配置（CONFIG、下载/解析函数等）。
如需改为完全自包含，请再告知以便内联这些工具实现。
"""
MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/bert-base-cased__train_val_revise.json__20251025_230335__exp1/best_hf"          # ✅ 例如 "bert-base-cased" 或本地路径
   
import os
import time
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
os.environ.pop("PYTORCH_INIT_ON_DEVICE", None)  # 保留一次即可
try:
    # 关键：若上游把 default_device 设成 'meta'，我们显式改回 CPU
    torch.set_default_device("cpu")
except Exception:
    pass

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # noqa: F401 (保留与原文件一致)
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch import nn
from collections import defaultdict
from pathlib import Path
# ================== 仍依赖的工具与配置（来自 utils.utils_loop） ==================
from utils.utils_loop import (
    CONFIG, logger as core_logger,
    ensure_dir, ledger_day_name, now_iso,
    JsonState, fetch_arxiv_recent, BertGate,
    download_pdf, grobid_extract_to_txt, txt_to_json,
    pdf_link_from_id,
    download_latex_source, extract_tar,
    parse_links_with_sentence_context,classify_link_inputs_via_llm,
    _FINAL_WRITE_LOCK,
    _build_single_window_around_pivot,
    _post_qwen_with_retry,     # ← 新增
)

os.environ.setdefault("NO_PROXY", "export.arxiv.org")
os.environ.setdefault("no_proxy", "export.arxiv.org")

from tqdm import tqdm
from utils.utils_train import (
     get_device,
    build_tokenizer, build_backbone_model, SentenceClassifier,
    make_dataloader
)
# 顶部 imports 补上：
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import threading, os, time as _t


# ========================================================================
# =                      全局配置 & 解析工具函数                           =
# ========================================================================

#——————————————————————LLM配置区————————————————————————————
'''
基础：
CONFIG.LLM_MODEL（模型名）
CONFIG.LLM_TEMPERATURE（温度）
CONFIG.DASHSCOPE_API_KEY_ENV（读取 API Key 的环境变量名；实际 key 请 export 到该变量）
长度/预算：
CONFIG.MAX_LLM_INPUT_CHARS（默认 50000，字符级预算） #写在本文件
CONFIG.LLM_OVERHEAD_RATIO（默认 0.9，用于留出 JSON/系统提示余量）#写在本文件
CONFIG.DESC_LLM_MAX_INPUT_CHARS（description 阶段每次调用的预算，默认 20000）
流程开关/策略：
CONFIG.DESC_SOURCE："bert" 或 "llm"（描述提取用 BERT 还是 LLM）
CONFIG.LATEX_LLM_DISABLED（可由环境变量 LATEX_LLM_DISABLED 控制；1=LaTeX 阶段禁用LLM，仅规则选链） #写在本文件
CONFIG.PRIMARY_HOST_TOPK、CONFIG.PREFERRED_DATASET_HOSTS（主链偏好）
CONFIG.FALLBACK_TOP1_ALWAYS、CONFIG.RULE_ONLY_ALWAYS_TOP1、CONFIG.DISABLE_PREFERRED_PICK（选链回退/偏好）
CONFIG.APPEND_LINK_SUMMARIES_TO_DESC（是否把安全摘要追加到 description）
'''

# —— description 路径（你已有，留作统一风格）description相关的 LLM 调用：——
CONFIG.DESC_SOURCE              = os.getenv("DESC_SOURCE", "llm").lower()  # llm|bert
CONFIG.DESC_LLM_MAX_INPUT_CHARS = int(os.getenv("DESC_LLM_MAX_INPUT_CHARS", "10000")) 
# —— link 路径——
CONFIG.LINK_CLASSIFIER          = os.getenv("LINK_CLASSIFIER", "rule").lower()   # rule|llm
CONFIG.LINK_PICKER              = os.getenv("LINK_PICKER", "hybrid").lower()     # rule|hybrid|llm
# —— 全局 LLM 预算 链接分类/打分相关的 LLM 调用：—— 
CONFIG.MAX_LLM_INPUT_CHARS      = int(os.getenv("MAX_LLM_INPUT_CHARS", "10000"))
CONFIG.LLM_OVERHEAD_RATIO       = float(os.getenv("LLM_OVERHEAD_RATIO", "0.9"))
DEFAULT_LLM_CHAR_LIMIT          = max(1000, int(CONFIG.MAX_LLM_INPUT_CHARS * CONFIG.LLM_OVERHEAD_RATIO))
# —— 可一键禁用 LaTeX 阶段的 LLM（保持你原有语义）——
CONFIG.LATEX_LLM_DISABLED       = bool(int(os.getenv("LATEX_LLM_DISABLED", "0")))
# CONFIG.LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# CONFIG.LLM_MODEL = "qwen3-235b-a22b-instruct-2507"
CONFIG.LLM_BASE_URL = "http://127.0.0.1:8000/v1"
CONFIG.LLM_MODEL = "QwQ-32B"
# 配置： PDF 并发下载
CONFIG.WORKERS = 1              # 总并发（整条流水线的并发）
CONFIG.PDF_MAX_CONCURRENCY = 4  # arXiv 下载并发上限
_STATE_LOCK = threading.Lock()
_PDF_SEM = threading.Semaphore(CONFIG.PDF_MAX_CONCURRENCY)

MIN_DESC_CHARS = 1
# --- Feature flags & safe defaults ---
# 开：把 LaTeX 阶段的 LLM 摘要（经清洗）回写到 dataset_description，利于检索
CONFIG.APPEND_LINK_SUMMARIES_TO_DESC = False 



# ---- arXiv API 简单节流（防 429 / 超时）----
_ARXIV_LAST = 0.0
_ARXIV_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL_S", "3.2"))  # 每次请求至少间隔 3.2 秒

def arxiv_throttle():
    """在调用 fetch_arxiv_recent 之前调一次，确保请求间隔"""
    global _ARXIV_LAST
    now = time.time()
    wait = _ARXIV_MIN_INTERVAL - (now - _ARXIV_LAST)
    if wait > 0:
        time.sleep(wait)
    _ARXIV_LAST = time.time()



#——————————————————————LLM配置区————————————————————————————

if not hasattr(CONFIG, "PRIMARY_HOST_TOPK"):
    CONFIG.PRIMARY_HOST_TOPK = 5  # 在前5名里先找“可信主机”

if not hasattr(CONFIG, "PREFERRED_DATASET_HOSTS"):
    CONFIG.PREFERRED_DATASET_HOSTS = [
        "huggingface.co/datasets",
        "zenodo.org/record",
        "kaggle.com/datasets",
        "figshare.com",
        "dataverse.org",
        "osf.io",
        "archive.org/details",
        "datahub.io",
        "storage.googleapis.com",
        "drive.google.com",
        "github.com/*/releases",
        "github.com/*/(blob|tree)/main/data",
        "raw.githubusercontent.com",
    ]
# ====== 新增/替换：更细的规则打分 ======
_HOST_WEIGHTS_POS = {
    "huggingface.co/datasets": 10,
    "zenodo.org/record": 9,
    "kaggle.com/datasets": 8,
    "figshare.com": 8,
    "dataverse.org": 8,
    "osf.io": 7,
    "archive.org/details": 6,
    "datahub.io": 6,
    "storage.googleapis.com": 5,
    "drive.google.com": 4,
    # GitHub 细粒度
    "github.com/": 0,  # 先给基线，下面再细分
    "github.com/*/releases": 6,
    "github.com/*/(blob|tree)/main/data": 5,
    "raw.githubusercontent.com": 3,
}
_HOST_WEIGHTS_NEG = {
    "arxiv.org": -10, "doi.org": -8, "dx.doi.org": -8, "acm.org": -8,
    "ieeexplore.ieee.org": -8, "scholar.google": -10,
    "researchgate.net": -6, "medium.com": -6,
}

_PATH_HINTS = {
    "/dataset": 3, "/datasets": 3, "/data": 2, "/download": 2,
    "/files": 2, "/record": 2, "/releases": 2,
    "/blob/main/data": 2, "/tree/main/data": 2,
}
_EXT_WEIGHTS = {
    ".csv": 6, ".tsv": 6, ".json": 6, ".parquet": 6,
    ".zip": 5, ".tar": 5, ".tar.gz": 5, ".tgz": 5, ".xz": 5, ".7z": 5, ".rar": 4
}
_POS_KEYS = ("dataset","data availability","we release","our dataset",
             "available at","we present","we introduce a dataset","benchmark","corpus")
_NEG_KEYS = ("code","source code","implementation","paper","supplementary material",
             "bibtex","project page","homepage","website")
_SPECIAL_NEG_PHRASES = (("we evaluate on","dataset"), ("we use","dataset"))

_LLM_SEM = threading.Semaphore(1)  # LLM 并发限流
# 全局会话（长连接 + 重试）
_SESSION = requests.Session()
_RETRY = Retry(
    total=5, connect=3, read=3,
    backoff_factor=2.0,                   # 指数退避更猛一点
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),   # GET 就够了
    respect_retry_after_header=True,      # 关键：遵守 Retry-After
    raise_on_status=False,                # 让它别立刻抛，交给 backoff
)
_ADAPTER = HTTPAdapter(max_retries=_RETRY, pool_connections=20, pool_maxsize=50)
_SESSION.mount("https://", _ADAPTER)


#========= 提示词构建（整篇 ≤100句） =========
def format_paper_for_prompt( paper: Dict, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
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
def label_sentences_via_llm(json_path: str, max_input_chars: Optional[int] = None) -> bool:
    """
    把整篇所有 section 的句子扁平化后做分块标注，再按映射写回各自原位。
    分块策略：按字符预算(max_input_chars 或 DEFAULT_LLM_CHAR_LIMIT)动态打包句子，避免超上下文。
    """
    with _LLM_SEM:
        try:
            # 读取原 JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data.get("data", []) or []
            if not items:
                return False

            # ---- 1) 扁平化 + 反向索引 ----
            flat: List[str] = []
            back: List[Tuple[int, int]] = []  # global_idx -> (section_idx, local_idx)
            for si, it in enumerate(items):
                for sj, s in enumerate(it.get("sentences", []) or []):
                    txt = s.get("text") or s.get("sentence_text") or ""
                    if txt and txt.strip():
                        back.append((si, sj))
                        flat.append(txt.strip())

            if not flat:
                return False

            paper_name = items[0].get("paper_name", "")
            budget = int(max_input_chars or DEFAULT_LLM_CHAR_LIMIT) 

            # ---- 2) 工具：模式、请求、解析 ----
            pat = re.compile(r"句子\s*(\d+)\s*[,，．.、]?\s*标记为\s*([01])", re.IGNORECASE)

            def call_once(prompt_text: str, timeout_sec: int = 90) -> str:
                payload = {
                    "model": CONFIG.LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是论文数据集描述识别助手。请严格按要求输出：仅逐行输出“句子X,标记为Y。解释：Z”。禁止输出其它内容。"
                        },
                        {"role": "user", "content": prompt_text}
                    ],
                    "temperature": CONFIG.LLM_TEMPERATURE,
                }
                data_resp = _post_qwen_with_retry(payload, timeout=(10, max(120, timeout_sec)), tries=3)
                return data_resp["choices"][0]["message"]["content"]

            def parse_and_collect(content: str, start_idx: int, end_idx: int, label_map: Dict[int, int]) -> int:
                ok = 0
                for line in content.splitlines():
                    m = pat.search(line)
                    if not m:
                        continue
                    gidx = int(m.group(1)) - 1  # 全局0-based
                    y = int(m.group(2))
                    if start_idx <= gidx < end_idx and y in (0, 1):
                        label_map[gidx] = y
                        ok += 1
                return ok

            # ---- 3) 预算切块生成器 ----
            def _truncate_line(s: str, cap: int = 480) -> str:
                s = (s or "").strip()
                return s if len(s) <= cap else s[:cap] + "..."

            def _iter_char_budget_chunks(flat_sentences: List[str], paper_name: str, start_from: int, cap: int):
                """
                逐句累加到 prompt 中，直到超过字符预算 cap；超过则回退到上一次可用切点。
                同时对单句过长做截断兜底。
                """
                n = len(flat_sentences)
                i = start_from
                while i < n:
                    j = i
                    last_good = None
                    while j < n:
                        lines = [{"text": _truncate_line(t)} for t in flat_sentences[i:j+1]]
                        prompt_try = build_chunk_prompt(paper_name, lines, start_idx=i)
                        if len(prompt_try) <= cap:
                            last_good = (j + 1, prompt_try)
                            j += 1
                            # 额外安全：限制每块句子上限，避免极端长标题/编号导致超大块
                            if (j - i) >= 160:
                                break
                        else:
                            break

                    if last_good is None:
                        # 单句也超预算：缩得更狠后强行提交一条
                        lines = [{"text": _truncate_line(flat_sentences[i], cap=max(200, cap // 8))}]
                        prompt_one = build_chunk_prompt(paper_name, lines, start_idx=i)
                        yield (i, i + 1, prompt_one)
                        i += 1
                    else:
                        end_idx, prompt_ok = last_good
                        yield (i, end_idx, prompt_ok)
                        i = end_idx

            # ---- 4) 分块打标（按字符预算）----
            label_map: Dict[int, int] = {}
            MAX_ATTEMPTS = int(getattr(CONFIG, "LLM_LABEL_MAX_ATTEMPTS", 3))  # 总尝试次数：含首发
            for b, e, prompt_text in _iter_char_budget_chunks(flat, paper_name, 0, budget):
                attempts = 0
                filled = 0
                while attempts < MAX_ATTEMPTS:
                    cur_prompt = (
                        prompt_text if attempts == 0
                        else "上一次输出的行数或编号不匹配。请仅按严格格式重新输出本块的全部标注：\n" + prompt_text
                    )
                    content = call_once(cur_prompt, timeout_sec=90)
                    parse_and_collect(content, b, e, label_map)
                    # 统计该块已成功写入的条数
                    filled = sum(1 for i in range(b, e) if i in label_map)
                    if filled == (e - b):
                        break
                    attempts += 1

                if filled != (e - b):
                    logger.warning("LLM 标注数量仍不匹配：%s / %s（块起始 %s，尝试 %d 次）",
                                   filled, (e - b), b, attempts)


            # ---- 5) 写回标签 ----
            for gidx, y in label_map.items():
                si, sj = back[gidx]
                try:
                    items[si]["sentences"][sj]["label"] = int(y)
                except Exception:
                    pass  # 防御

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True

        except Exception as e:
            logger.exception("label_sentences_via_llm error: %s", e)
            return False




# ====  LLM：从全文抽“数据集链接” ====
def extract_dataset_links_via_llm(txt_path:str)->List[str]:
    with _LLM_SEM:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                paper_text = f.read()
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

   
def _truncate_for_llm(input_text: str, limit: int) -> str:
    if not isinstance(input_text, str):
        return str(input_text)
    text = input_text.strip()
    if len(text) <= limit:
        return text

    # 优先保留 [URL] + [ANCHOR] 段；对 [CONTEXT] 做有损裁剪
    m = re.match(r"(?s)^(\[URL\].*?\s*\[ANCHOR\].*?\s*\[CONTEXT\]\s*)", text)
    if m:
        head = m.group(1)
        body = text[len(head):]
    else:
        head, body = "", text

    budget = max(0, limit - len(head) - 3)
    body = body[:budget].rstrip()
    return head + body + "..."    
    
def _pick_by_preferred_hosts(cands, topk=None):
    """从前 topk 名里，优先挑命中“数据集托管主机”的；并做轻量 tie-break。"""
    topk = int(topk or getattr(CONFIG, "PRIMARY_HOST_TOPK", 5))
    ranked = sorted(cands, key=lambda x: x["score"], reverse=True)[:topk]
    pats = list(getattr(CONFIG, "PREFERRED_DATASET_HOSTS", []))

    def _host_match(u: str) -> bool:
        hp = _host_path(_norm_url(u))
        for pat in pats:
            if pat == "github.com/*/releases":
                if "github.com" in hp and "/releases" in hp: return True
            elif pat == "github.com/*/(blob|tree)/main/data":
                if "github.com" in hp and ("/blob/main/data" in hp or "/tree/main/data" in hp): return True
            else:
                if pat in hp: return True
        return False

    preferred = [c for c in ranked if _host_match(c["url"])]
    if not preferred:
        return None

    # 细化：同分时优先“落地页”而非直接文件（更稳）
    def _is_landing(u: str) -> int:
        return 1 if not any(_norm_url(u).endswith(ext) for ext in _EXT_WEIGHTS.keys()) else 0

    preferred.sort(key=lambda c: (c["score"], _is_landing(c["url"])), reverse=True)
    b = preferred[0]
    return {"url": b["url"], "why": b.get("anchor", ""), "primary_score": float(b.get("score", 0))}

def _start_speed_monitor(filepath: str, tag: str = "NET"):
    stop = threading.Event()
    def _run():
        last_sz, last_t = 0, _t.time()
        while not stop.is_set():
            _t.sleep(1.0)
            try:
                sz = os.path.getsize(filepath)
            except Exception:
                continue
            now = _t.time()
            if now == last_t: 
                continue
            inst_mbps = (sz - last_sz) * 8.0 / (now - last_t) / 1e6
            core_logger.info("[NET][%s] %s size=%.2f MB, speed=%.2f Mbps",
                             tag, os.path.basename(filepath), sz / 1e6, inst_mbps)
            last_sz, last_t = sz, now
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop

def _score_candidate_v2(url: str, anchor: str, ctx: str) -> Tuple[int, Dict]:
    u_norm = _norm_url(url)
    hp = _host_path(u_norm)
    score, reasons = 0, []

    # Host 权重
    def _add(reason, pts): 
        nonlocal score; score += pts; reasons.append(f"{reason}:{pts}")

    # 正向 host
    for pat, w in _HOST_WEIGHTS_POS.items():
        if pat.endswith("/"):
            if hp.startswith(pat): _add(f"host+:{pat}", w)
        elif pat == "github.com/*/releases":
            if "github.com" in hp and "/releases" in hp: _add("host+:gh_releases", w)
        elif pat == "github.com/*/(blob|tree)/main/data":
            if "github.com" in hp and ("/blob/main/data" in hp or "/tree/main/data" in hp):
                _add("host+:gh_data_dir", w)
        else:
            if pat in hp: _add(f"host+:{pat}", w)

    # 负向 host
    for pat, w in _HOST_WEIGHTS_NEG.items():
        if pat in hp: _add(f"host-:{pat}", w)

    # GitHub 仓库根目录惩罚
    if "github.com" in hp and not ("/releases" in hp or "/blob/main/data" in hp or "/tree/main/data" in hp or "/data/" in hp):
        _add("gh_repo_root", -4)

    # Path hints
    for ph, w in _PATH_HINTS.items():
        if ph in hp: _add(f"path:{ph}", w)

    # 扩展名
    for ext, w in _EXT_WEIGHTS.items():
        if u_norm.endswith(ext):
            _add(f"ext:{ext}", w)
            break

    # 文本关键词（合并 anchor+ctx）
    text = f"{anchor or ''} {ctx or ''}".lower()
    pos_hits = sum(1 for k in _POS_KEYS if k in text)
    neg_hits = sum(1 for k in _NEG_KEYS if k in text)
    if pos_hits:
        _add("kw_pos", min(8, pos_hits * 2))   # 每命中 +2，上限 +8
    if neg_hits:
        _add("kw_neg", -min(6, neg_hits * 2))  # 每命中 -2，下限 -6
    for a, b in _SPECIAL_NEG_PHRASES:
        if a in text and b in text:
            _add("kw_neg_phrase", -3)

    return score, {"reasons": reasons}

# ====== 新增：仅用规则直接挑主（失败再交给 LLM） ======
def _choose_primary_rule_only(cands: List[Dict]) -> Optional[Dict]:
    """
    cands: 每项至少包含 {url, anchor, ctx, score}
    返回 {url, why, primary_score} 或 None
    """
    if not cands: return None
    cands = sorted(cands, key=lambda x: x.get("score", 0), reverse=True)
    if len(cands) == 1:
        best = cands[0]
        return {"url": best["url"], "why": best.get("anchor",""), "primary_score": float(best["score"])}

    s1, s2 = cands[0]["score"], cands[1]["score"]
    # 直出阈值
    if s1 >= 22 or (s1 >= 16 and (s1 - s2) >= 5):
        best = cands[0]
        return {"url": best["url"], "why": best.get("anchor",""), "primary_score": float(best["score"])}

    # 平局破除（只看前5名）
    def _host_priority(hp: str) -> int:
        if "huggingface.co/datasets" in hp: return 5
        if any(p in hp for p in ("zenodo.org/record","kaggle.com/datasets","figshare.com","dataverse.org")): return 4
        if "github.com" in hp and ("/releases" in hp or "/blob/main/data" in hp or "/tree/main/data" in hp): return 3
        if "storage.googleapis.com" in hp: return 2
        if "drive.google.com" in hp: return 1
        return 0

    def _is_landing_page(u: str) -> int:
        # 落地页优先（不是直接文件）
        return 1 if not any(u.lower().endswith(ext) for ext in _EXT_WEIGHTS) else 0

    def _pri_key(c):
        hp = _host_path(_norm_url(c["url"]))
        return (
            _host_priority(hp),
            _is_landing_page(c["url"]),
            c["score"],
            -len(c["url"])  # 更短优先
        )

    best = max(cands[:5], key=_pri_key)
    if best["score"] >= 15:
        return {"url": best["url"], "why": best.get("anchor",""), "primary_score": float(best["score"])}
    return None


# utils/utils_loop.py -> class CONFIG:
# ==== LaTeX 窗口上下文注入策略 ====
logger = logging.getLogger("realtime")

# --- Prompt/Log/Code 污染检测（用于清洗）---
_PROMPT_LEAK = re.compile(
    r'(\bEVIDENCE\b|\bOUTPUT\s*FORMAT\b|\{context\}|\bTool Response\b|\bAgent\b|\bAPI[-\s]|```|^>>|^#+\s)',
    re.IGNORECASE | re.MULTILINE
)
    
def _looks_like_code_or_log(t: str) -> bool:
    s = (t or "").strip()
    if not s:
        return False
    if _PROMPT_LEAK.search(s):
        return True
    if re.search(r'^\s*[{[][^\n]*[]}]', s):   # 一整行 JSON/数组
        return True
    if re.search(r'^[A-Z ][A-Z0-9 \-:_]{12,}$', s):  # 模板/标题行
        return True
    # 代码/符号密度
    sym = sum(ch in "{}[]()<>=_`#*~^$\\|;" for ch in s)
    if len(s) >= 48 and sym / len(s) > 0.14:
        return True
    return False


_SENT_PUNCTS = "。！？.!?；;"
from urllib.parse import urlsplit
_URL_RE = re.compile(r'https?://[^\s)\]]+')

def _sanitize_llm_summary_line(s: str, max_len: Optional[int] = None) -> str:
    """仅做最小清洗；删除 URL；不做最短长度门槛；可选不截断。"""
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # 去掉“提示词/日志/代码”痕迹
    s = re.sub(r"(Agent:|Tool Response:|System:|User:|Assistant:)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)
    if _looks_like_code_or_log(s):
        return ""
    # 直接移除 URL（不再替换为域名）
    s = _URL_RE.sub("", s).strip()
    # 不再做 <18 字符的丢弃
    if max_len and max_len > 0 and len(s) > max_len:
        s = s[:max_len].rstrip() + " ..."
    return s




# 路径配置
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def _find_bracket_span(s: str):
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return start, i + 1
    return None


def _safe_json_loads(raw: str) -> Any:
    """尽量把 LLM 的“近似 JSON”解析成对象；支持代码块、JSON Lines、粘连 JSON。"""
    if not isinstance(raw, str):
        return raw
    txt = _strip_code_fences(raw)

    # 直接解析
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    # JSON Lines：每行一个对象
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines and all(ln.startswith("{") for ln in lines):
        objs = []
        ok = True
        for ln in lines:
            try:
                objs.append(json.loads(ln))
            except Exception:
                ok = False
                break
        if ok:
            return objs

    # 提取首个完整 {...} 或 [...]
    span = _find_bracket_span(txt)
    if span:
        s, e = span
        core = txt[s:e]
        try:
            return json.loads(core)
        except Exception:
            pass

    # 处理粘连 {"a":1}{"b":2}
    parts = re.split(r"}\s*{", txt.strip())
    if len(parts) > 1:
        joined = "{" + "},{".join(p.strip("{} \n\t\r") for p in parts) + "}"
        return json.loads(f"[{joined}]")  # 让异常抛出以暴露真实问题

    # 最后再试一次：取最后一行
    last = lines[-1] if lines else txt
    return json.loads(last)  # 失败就抛，外层捕获并降级为 other

# ========================================================================
# =                           BERT 句级打标主流程                         =
# ========================================================================

def _log_description(arxiv_id: str, dataset_desc: str, json_path: str) -> None:
    """打印是否抽到描述、长度与预览（避免日志过长，截断到120字符）"""
    if dataset_desc and dataset_desc.strip():
        preview = re.sub(r"\s+", " ", dataset_desc[:120])
        core_logger.info("[Desc] %s ✓ extracted (%d chars) from %s | preview: %s",
                         arxiv_id, len(dataset_desc), os.path.basename(json_path), preview)
    else:
        core_logger.warning("[Desc] %s ✗ empty after BERT labels (json=%s)",
                            arxiv_id, os.path.basename(json_path))

def load_json_papers(json_file: str) -> List[Dict]:
    """读取你们的数据格式，并按 paper_name 聚合 sentences。
    期望输入结构：{"data":[{paper_name, section, paragraph_id, sentences:[{text,label},...]}, ...]}
    输出：[{"paper_name": str, "sentences": List[{"text":str,"label":int}]}]
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw['data']
    papers = []
    current = None
    for item in sorted(data, key=lambda x: (x['paper_name'], x['section'], x['paragraph_id'])):
        if current is None or item['paper_name'] != current['paper_name']:
            if current is not None:
                papers.append(current)
            current = {"paper_name": item['paper_name'], "sentences": []}
        current['sentences'].extend(item['sentences'])
    if current is not None:
        papers.append(current)
    return papers


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    纯推理版 evaluate：
    - 不传 labels、不计算 loss/metrics；
    - 仅前向得到 logits -> 收集成结构化预测；
    - metrics 用占位的 0 值返回，保持接口兼容。
    """
    model.eval()

    predictions_data: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # 前向：明确不传 labels
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            # 兼容返回形式： (None, logits) 或 logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[-1]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=-1)          # [B, S, 2]
            preds = torch.argmax(logits, dim=-1)           # [B, S]

            #原本的Collectpredictions的部分: indices的来源
            indices      = batch['indices']                # [B, S]（-1 为 padding）
            paper_names  = batch['paper_names']            # List[str]
            sentences_ls = batch['sentences']              # List[List[str]]

            B = len(paper_names)
            
            
            for b in range(B):
               # 新逻辑（稳健对齐）
                seq_len = preds.shape[1]                        # 模型输出的实际序列长度
                idx_row = indices[b][:seq_len]                  # 用输出长度裁剪 indices
                pred_row = preds[b][:seq_len]
                prob_row = probs[b][:seq_len]

                valid_mask    = (idx_row != -1)                 # 只在裁剪后做掩码
                valid_indices = idx_row[valid_mask]
                valid_preds   = pred_row[valid_mask]
                valid_probs   = prob_row[valid_mask]
                
                # 搬回cpu
                valid_indices = valid_indices.cpu()
                valid_preds   = valid_preds.cpu()
                valid_probs   = valid_probs.cpu()
                # 注意 sentences 对齐：截到同样的有效长度
                valid_sentences = sentences_ls[b][:len(valid_indices)]

                sent_items = []
                for idx_t, yhat_t, pvec, sent_text in zip(
                    valid_indices, valid_preds, valid_probs, valid_sentences
                ):
                    sid  = int(idx_t.item())
                    yhat = int(yhat_t.item())
                    sent_items.append({
                        "sentence_id":    sid,
                        "idx":            sid,           # 兼容你的聚合逻辑
                        "sentence_text":  sent_text,
                        "label":          yhat,
                        "y":              yhat,
                        "true_label":     -100,          # 明确无标签
                        "confidence":     float(pvec[yhat].item()),
                        "probabilities": {
                            "class_0": float(pvec[0].item()),
                            "class_1": float(pvec[1].item()),
                        },
                    })

                if sent_items:
                    predictions_data.append({
                        "paper_name": paper_names[b],
                        "sentences":  sent_items,
                    })

    # 纯推理占位指标（保持返回结构不变）
    metrics = {"acc": 0.0, "prec": 0.0, "recall": 0.0, "f1": 0.0, "loss": 0.0}
    return metrics, predictions_data


def collect_dataset_description(json_path: str, max_chars: Optional[int] = None) -> str:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pos_texts = []
        for it in data.get("data", []) or []:
            for s in it.get("sentences", []) or []:
                lab = int(s.get("label", s.get("predicted_label", 0)) or 0)
                txt = s.get("text") or s.get("sentence_text") or ""
                if lab == 1 and txt.strip():
                    pos_texts.append(txt.strip())

        desc = " ".join(pos_texts).strip()
        if not desc:
            return ""

        if isinstance(max_chars, int) and max_chars > 0 and len(desc) > max_chars:
            return desc[:max_chars].rstrip() + " ..."
        return desc
    except Exception:
        return ""
    
    
def label_sentences_via_bert(
    json_path: str,
    context_size: int = 2,
    max_length: int = 512,
    threshold: Optional[float] = None,  # 保留但不使用；与你现有评估保持一致
    min_sentences: int = 3
) -> bool:
    """
    与 train.py 的评估路径保持一致：
      - 读入 JSON（需符合 load_json_papers 期望的结构）
      - 构建 tokenizer/backbone/SentenceClassifier
      - 尝试加载 classifier_head.bin（无“概率均值”聚合）
      - make_dataloader -> evaluate
      - 保存 preds 到保存目录（不改写原 JSON）
    返回：是否成功
    """
    try:
        # -------- 1) 数据加载 --------
        val_papers = load_json_papers(json_path)
        # val_papers = load_json_papers("/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json") #测试：采用ｔｅｓｔ数据集，看看是不是数据集问题
        
        
        if not val_papers:
            print(f"val_papers空数据：{json_path}")
            return False
        print(f"[label] Loaded {len(val_papers)} papers from {os.path.basename(json_path)}")

        # -------- 2) 模型与 tokenizer 初始化（与 train.py 一致）--------
        #2.1/2.2/2.3  tokenizer，backbone，model加载；
        model_src = MODEL_PATH
        tokenizer = build_tokenizer(model_src)
        backbone  = build_backbone_model(model_src)
        model     = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2)

        #2.4 head加载
        head_path = os.path.join(model_src, "classifier_head.bin")
        state = torch.load(head_path, map_location="cpu")
        
        try:
            incomp = model.classifier.load_state_dict(state, strict=False)
            if incomp.missing_keys or incomp.unexpected_keys:
                print(f"[label] head key mismatch: missing={incomp.missing_keys}, unexpected={incomp.unexpected_keys}")
        except RuntimeError:
            # 兼容 'module.' 前缀
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            incomp = model.classifier.load_state_dict(state, strict=False)  
        print(f"[label] classifier_head.bin loaded from {head_path}")
        
        # 3) 移动模型
        device = get_device()
        model  = model.to(device)
 
        # 4) 构建 DataLoader ： 需要检查：每次爬取得到的pdf被解析成json之后，是不是传入的这个路径
        # 与训练/评估一致；不做任何额外聚合
        val_loader = make_dataloader(
            val_papers,
            tokenizer,
            batch_size=1,
            max_length=max_length,
            min_sentences=min_sentences,
            context_size=context_size,
            shuffle=False
        )

        # 5) 开始推理（保持你的评估逻辑不变）
        _, preds = evaluate(model, val_loader, device)


        # === ①读取preds===
        agg = defaultdict(dict)  # {paper_name: {sid: label}}
        for win in preds:#preds是一个列表，里面元素是字典
            p = win.get("paper_name")
            sent_list = win.get("sentences", []) #[{'sentence_id': 60, 'idx': 60, 'sentence_text': 'By enabling large-scale pre-training of LLMs on executable code and software artifacts, SBAN supports research in program understanding, malware detection, and secure software engineering.', 'label': 1, 'y': 1, 'true_label': -100, 'confidence': 0.9967065453529358, 'probabilities': {...}}, {'sentence_id': 61, 'idx': 61, 'sentence_text': 'We also believe there is still room for improvement to increase the diversity of samples in different programming languages, along with their associated binaries or assembly code, as part of future work.', 'label': 1, 'y': 1, 'true_label': -100, 'confidence': 0.9478492736816406, 'probabilities': {...}}, {'sentence_id': 62, 'idx': 62, 'sentence_text': 'DATASET BE USED?', 'label': 1, 'y': 1, 'true_label': -100, 'confidence': 0.9209175109863281, 'probabilities': {...}}]
            if not p or not isinstance(sent_list, list):
                continue
            for s in sent_list:
                sid = s.get("sentence_id", s.get("idx"))
                lab = s.get("label", s.get("y"))
                if sid is None or lab is None:
                    continue
                sid = int(sid) #2
                lab = int(lab) #1
                # 只要任一窗口预测为 1，就置 1；否则保留已有值（默认 0）
                agg[p][sid] = 1 if (lab == 1 or agg[p].get(sid, 0) == 1) else 0

        # === ②回写到 json_path 原文件 ===
        jp = Path(json_path)
        with jp.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # 先把同一 paper 的所有 section 收到一起
        paper2items = defaultdict(list)
        for it in raw.get("data", []):
            paper2items[it.get("paper_name", "")].append(it)

        # 使用和构建数据集一致的排序规则，确保顺序可重现
        def _sort_key(x):
            # 若你有更明确的排序（比如创建数据集时就是按 (section, paragraph_id)），保持一致
            return (x.get("section", ""), x.get("paragraph_id", 0))

        updated = 0
        missed  = 0

        for pname, items in paper2items.items():
            if not pname or pname not in agg:
                continue
            idx2y = agg[pname]     # 这里的 key 是“全局句子索引”
            # 关键：把这篇论文的所有段落按固定顺序串起来，然后累加全局索引
            items_sorted = sorted(items, key=_sort_key)

            global_sid = 0  # 全局索引，从 0 开始贯穿整篇论文
            for it in items_sorted:
                sents = it.get("sentences", [])
                for s in sents:
                    # 用全局索引对 agg，而不是用局部 sentence_id
                    if global_sid in idx2y:
                        s["label"] = int(idx2y[global_sid])
                        updated += 1
                    else:
                        # 没在预测里就保留原值（通常是 0）
                        s["label"] = int(s.get("label", 0))
                        missed += 1

                    # （可选）为排查方便，把全局索引也写回去
                    s["global_id"] = global_sid
                    global_sid += 1

        with jp.open("w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        print(f"[label] wrote predicted labels back to {jp} (updated={updated}, missed={missed})")

        # === 生成并保存最终 description（不保存中间预测文件）===
        desc = collect_dataset_description(str(jp))
        desc_path = jp.parent / (jp.stem + ".bert.desc.txt")
        with desc_path.open("w", encoding="utf-8") as f:
            f.write(desc or "")
        print(f"[label] description saved -> {desc_path}")

        return True
    
    except Exception as e:
        print(f"[label][ERROR] {os.path.basename(json_path)}: {e}")
        return False

# ========================================================================
# =                             日志初始化                                =
# ========================================================================

def setup_logging():
    """在运行脚本里配置日志 Handler，避免被多处导入时重复配置。"""
    log_dir = os.path.join(CONFIG.BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "realtime.log")

    core_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)

    # 避免重复添加
    core_logger.propagate = False  # 关键：不要向 root 传播，避免重复打印
    have_sh = any(isinstance(h, logging.StreamHandler) for h in core_logger.handlers)
    have_fh = any(isinstance(h, logging.FileHandler) for h in core_logger.handlers)
    if not have_sh: core_logger.addHandler(sh)
    if not have_fh: core_logger.addHandler(fh)


# ========================================================================
# =                          PDF → 描述抽取流水线                          =
# ========================================================================

def process_until_description(cat: str, entry: dict) -> dict:
    """
    处理单篇论文：下载 PDF -> GROBID -> TXT -> JSON -> BERT 句子贴标签 -> 汇总 data_description
    不做链接抽取与最终四列表。返回阶段性产物字典。
    """
    arxiv_id = entry["id"]
    day = ledger_day_name()

    # 目录与路径
    pdf_dir = os.path.join(CONFIG.OUT_PDF, day, cat)
    ensure_dir(pdf_dir)
    txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
    json_path = os.path.join(pdf_dir, f"{arxiv_id}.json")
    pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")

    # Ⅰ 下载 PDF（若已存在则跳过）：所以之前运行过一遍之后，再运行，PDF不会重复下载 01:28 pdf_download = 267081 ms    13.49mB 
    if not os.path.exists(pdf_path):
        mon = _start_speed_monitor(pdf_path, tag=f"PDF:{arxiv_id}")
        try:
            # 限制同时对 arXiv 发起的下载数量
            with _PDF_SEM:
                if not download_pdf(pdf_link_from_id(arxiv_id), pdf_path):
                    raise RuntimeError(f"PDF download failed for {arxiv_id}")
        finally:
            mon.set()
    # Ⅱ GROBID -> TXT;
    if not os.path.exists(txt_path):
        if not grobid_extract_to_txt(pdf_path, txt_path):
            raise RuntimeError(f"GROBID failed for {arxiv_id}")

    # Ⅲ TXT -> JSON（句子切分，默认 label=0）
    if not os.path.exists(json_path):
        if not txt_to_json(txt_path, json_path):
            raise RuntimeError(f"txt_to_json failed for {arxiv_id}")

    # Ⅳ 句级打标（可选 BERT / LLM）
    desc_source = (getattr(CONFIG, "DESC_SOURCE", "llm") or "llm").lower()
    if desc_source == "llm":
        ok = label_sentences_via_llm(
            json_path=json_path,
            max_input_chars=int(getattr(CONFIG, "DESC_LLM_MAX_INPUT_CHARS", 20000))
        )
        core_logger.info("[DescMode] %s use LLM (max_input=%s) -> %s",
                         arxiv_id, getattr(CONFIG, "DESC_LLM_MAX_INPUT_CHARS", 20000), ok)
    else:
        ok = label_sentences_via_bert(
            json_path=json_path,
            context_size=int(getattr(CONFIG, "SENT_CTX_SIZE", 2)),
            max_length=int(getattr(CONFIG, "SENT_MAX_LEN", 512)),
            threshold=float(getattr(CONFIG, "BERT_SENT_THRESHOLD", getattr(CONFIG, "BERT_THRESHOLD", 0.5)))
        )
        core_logger.info("[DescMode] %s use BERT -> %s", arxiv_id, ok)

    if not ok:
        raise RuntimeError(f"sentence labeling failed for {arxiv_id}")

    # ---- 简单门控：仅按描述长度 ----
    # 生成描述，然后仅按字符长度做门控
    dataset_desc = collect_dataset_description(json_path)
    desc_len = len(dataset_desc or "")
        
    if desc_len < MIN_DESC_CHARS:
        core_logger.info(
            "[DescGate] %s REJECT: desc_len=%d (<%d)",
            arxiv_id, desc_len, MIN_DESC_CHARS
        )
        # 写阶段性产物，标注为未通过（不触发 LLM，不进 LaTeX）
        stage_dir = os.path.join(CONFIG.OUT_PROCESSED, day, cat)
        ensure_dir(stage_dir)
        stage_record = {
            "arxiv_id": arxiv_id,
            "category": cat,
            "paper_link_pdf": pdf_link_from_id(arxiv_id),
            "paths": {"pdf": pdf_path, "txt": txt_path, "sent_json": json_path},
            "dataset_description": dataset_desc,
            "title": entry.get("title", ""),
            "abstract": entry.get("abstract", ""),
            "updated": entry.get("updated", ""),
            "published": entry.get("published", ""),
            "bert_gate": {
                "prob": float(entry.get("_bert_prob", -1.0)),
                "threshold": float(CONFIG.BERT_THRESHOLD),
                "accepted": bool(float(entry.get("_bert_prob", -1.0)) >= float(CONFIG.BERT_THRESHOLD))
            },
            "desc_gate": {
                "accepted": False,
                "desc_len": desc_len,
                "min_desc_chars": MIN_DESC_CHARS,
            }
        }

        stage_path = os.path.join(stage_dir, f"{arxiv_id}.partial.json")
        with open(stage_path, "w", encoding="utf-8") as f:
            json.dump(stage_record, f, ensure_ascii=False, indent=2)
        return stage_record  # ← 直接返回，后续不做 LLM 兜底
    
    _log_description(arxiv_id, dataset_desc, json_path)

    # 写一个“阶段性产物”到 OUT_PROCESSED，便于后续 link_extraction 新方案继续接力
    stage_dir = os.path.join(CONFIG.OUT_PROCESSED, day, cat)
    ensure_dir(stage_dir)

    stage_record = {
        "arxiv_id": arxiv_id,
        "category": cat,
        "paper_link_pdf": pdf_link_from_id(arxiv_id),
        "paths": {"pdf": pdf_path, "txt": txt_path, "sent_json": json_path},
        "dataset_description": dataset_desc,
        "title": entry.get("title", ""),
        "abstract": entry.get("abstract", ""),
        "updated": entry.get("updated", ""),
        "published": entry.get("published", ""),
        "bert_gate": {
            "prob": float(entry.get("_bert_prob", -1.0)),
            "threshold": float(CONFIG.BERT_THRESHOLD),
            "accepted": bool(float(entry.get("_bert_prob", -1.0)) >= float(CONFIG.BERT_THRESHOLD))
        },
        "desc_gate": {
            "accepted": True,
            "desc_len": desc_len,
            "min_desc_chars": MIN_DESC_CHARS,
        }
    }

    core_logger.info(
        "[BERT] %s prob=%.3f thr=%.3f accepted=%s",
        arxiv_id,
        stage_record["bert_gate"]["prob"],
        stage_record["bert_gate"]["threshold"],
        stage_record["bert_gate"]["accepted"]
    )

    stage_path = os.path.join(stage_dir, f"{arxiv_id}.partial.json")
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump(stage_record, f, ensure_ascii=False, indent=2)

    return stage_record



def append_final_records(records: List[Dict]):
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

        merged = (records or []) + (existing or [])

        seen, dedup = set(), []
        for r in merged:
            paper = (r.get("Paper Link") or "").strip()
            dlink = (r.get("Dataset Link") or "").strip()
            if not paper or not dlink:
                continue
            # 关键：使用“归一化后的 URL”作为去重键
            key = (_norm_url(paper), _norm_url(dlink))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dedup, f, ensure_ascii=False, indent=2)

    logger.info("Final records written: %s (+%d)", out_path, len(records))


def _is_structural_dataset_url(url_norm: str) -> bool:
    """只有“像数据集”的 URL 才最终入库/回写摘要。"""
    hp = _host_path(url_norm)

    # 论文/索引/新闻/匿名页等直接拒
    if any(neg in hp for neg in _NEG_HOSTS):
        return False

    if any(h in hp for h in _DATA_HOST_HINTS):
        return True
    if url_norm.endswith(_DATA_EXTS):
        return True

    if "github.com" in hp and ("/releases" in hp or "/blob/main/data" in hp or "/tree/main/data" in hp or "/data/" in hp):
        return True
    if "kaggle.com" in hp and "/datasets/" in hp:
        return True
    return False



# ========================================================================
# =                       LLM 链接分类（稳健包装）                         =
# ========================================================================


def _call_llm_and_get_text(batch_inputs: List[Dict], max_input_length: Optional[int] = None) -> str:
    cap = int(max_input_length or DEFAULT_LLM_CHAR_LIMIT)
    safe_batch = []
    for item in batch_inputs:
        inp = item.get("input", "")
        inp = _truncate_for_llm(inp, cap)  # ← 只在发给 LLM 前截断
        safe_batch.append({**item, "input": inp})
    obj = classify_link_inputs_via_llm(safe_batch)
    return json.dumps(obj, ensure_ascii=False)





def classify_link_inputs_via_llm_safe(qwen_inputs: List[Dict], batch_size: int = 4, max_input_length: Optional[int] = None) -> List[Dict]:
    results = []
    for s in range(0, len(qwen_inputs), batch_size):
        e = min(len(qwen_inputs), s + batch_size)
        batch = qwen_inputs[s:e]
        try:
            raw = _call_llm_and_get_text(batch, max_input_length)
            obj = _safe_json_loads(raw)
            if isinstance(obj, list):
                parsed = obj
            elif isinstance(obj, dict) and "items" in obj:
                parsed = obj["items"]
            else:
                parsed = obj if isinstance(obj, list) else [obj]
        except Exception as ex:
            dump_dir = os.path.join(CONFIG.BASE_DIR, "logs", "llm_raw")
            ensure_dir(dump_dir)
            dump_path = os.path.join(dump_dir, f"batch_{s}_{e-1}.txt")
            try:
                with open(dump_path, "w", encoding="utf-8") as f:
                    f.write(raw if 'raw' in locals() else "<no raw>")
            except Exception:
                pass
            logger.error("classify_link_inputs_via_llm error on batch %d-%d: %s | dumped=%s", s, e-1, ex, dump_path)
            parsed = [{"url": x.get("url",""), "label": "other", "why": "parser_error"} for x in batch]

        results.extend(parsed)
    return results



# ========================================================================
# =                           LaTeX 链接抽取阶段                          =
# ========================================================================

# ==== 新增：URL 归一化 & 文本清洗 & 打分与摘要工具 ====
from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

# 强信号域/路径/扩展名
_DATA_HOST_HINTS = (
    "huggingface.co/datasets",
    "zenodo.org/record",
    "figshare.com",
    "osf.io",
    "kaggle.com/datasets",
    "dataverse.org",
    "datahub.io",
    "archive.org/details",
    "drive.google.com",
    "storage.googleapis.com",
    "opendatasoft.com",
)
_DATA_PATH_HINTS = (
    "/dataset", "/datasets", "/download", "/files", "/record",
    "/releases", "/data", "/blob/main/data", "/tree/main/data"
)
_DATA_EXTS = (".zip",".tar",".tar.gz",".tgz",".xz",".7z",".rar",".csv",".tsv",".json",".parquet")

_NEG_HOSTS = (
    "arxiv.org", "scholar.google", "dx.doi.org", "doi.org",
    "acm.org", "ieeexplore.ieee.org",
    # 常见非数据集/资讯/匿名镜像
    "statista.com", "tvisioninsights.com", "4open.science",
    "medium.com", "researchgate.net"
)

def _norm_url(u: str) -> str:
    if not u:
        return ""
    try:
        s = urlsplit(u.strip())
        # 忽略 scheme（http/https 统一），host/path 小写、去尾斜杠；去掉 utm_*；去掉 fragment
        netloc = (s.netloc or "").lower()
        path = (s.path or "").lower().rstrip("/")
        qs = [(k, v) for k, v in parse_qsl(s.query, keep_blank_values=True)
              if not k.lower().startswith("utm_")]
        query = urlencode(qs, doseq=True)
        # 关键：scheme 置空，fragment 丢弃
        return urlunsplit(("", netloc, path, query, ""))
    except Exception:
        return (u or "").strip().rstrip("/").lower()

def _host_path(u: str) -> str:
    try:
        s = urlsplit(u)
        return f"{s.netloc}{s.path}".lower()
    except Exception:
        return u.lower()

# --- LaTeX → 文本（强清洗，只留自然语言） ---
_RE_ENV_BLOCKS = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|figure\*?|table\*?)\}.*?\\end\{\1\}",
    re.DOTALL | re.IGNORECASE
)
_RE_INLINE_MATH = re.compile(r"(\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\])")
_RE_CITE_REF    = re.compile(r"\\(cite|Cite|ref|eqref|autoref|cref|Cref)\s*\{[^}]*\}")
_RE_CMD_ARG     = re.compile(r"\\(emph|textbf|textit|mathbf|mathrm|textrm|texttt|underline)\s*\{([^}]*)\}")
_RE_CMD_NOARG   = re.compile(r"\\(url|path|footnote)\s*\{[^}]*\}")
_RE_INPUT_INC   = re.compile(r"\\(input|includegraphics)\s*\{[^}]*\}")
_RE_BRACES      = re.compile(r"[{}]")
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_HTTP        = re.compile(r"(https?://\S+|doi:\S+)", re.IGNORECASE)

def _latex_to_text(s: str) -> str:
    if not s: return ""
    s = _RE_ENV_BLOCKS.sub(" ", s)
    s = _RE_INLINE_MATH.sub(" ", s)
    s = _RE_CITE_REF.sub(" ", s)
    s = _RE_CMD_NOARG.sub(" ", s)
    s = _RE_INPUT_INC.sub(" ", s)
    s = _RE_CMD_ARG.sub(lambda m: m.group(2), s)
    s = _RE_HTTP.sub(" ", s)
    s = _RE_BRACES.sub("", s)
    repl = {r"\%":"%", r"\_":"_", r"\&":"&", r"\#":"#", r"``":"“", r"''":"”", r"---":"—", r"--":"–", r"~":" "}
    for k, v in repl.items():
        s = s.replace(k, v)
    s = _RE_MULTI_SPACE.sub(" ", s).strip()
    # 自然语言性过滤：长度 + 符号密度
    letters = sum(ch.isalnum() for ch in s)
    if len(s) < 12 or (len(s) - letters) / max(len(s), 1) > 0.38:
        return ""
    return s

# --- 可解释打分：规则为主 ---
def _score_candidate(url_norm: str, anchors: list, contexts: list):
    hp = _host_path(url_norm)
    score, reasons = 0, []

    if any(h in hp for h in _DATA_HOST_HINTS):
        score += 5; reasons.append("host_hint")
    if any(p in hp for p in _DATA_PATH_HINTS):
        score += 3; reasons.append("path_hint")
    if url_norm.endswith(_DATA_EXTS):
        score += 3; reasons.append("file_ext")

    if any(h in hp for h in _NEG_HOSTS):
        score -= 3; reasons.append("neg_host")
    if "github.com" in hp and not ("/releases" in hp or "/data" in hp or "/blob/main/data" in hp or "/tree/main/data" in hp):
        score -= 3; reasons.append("github_repo_root")

    ctx = " ".join((anchors or []) + (contexts or [])).lower()
    pos_keys = ("dataset", "data availability", "download", "corpus", "benchmark", "we release", "available at", "provided at")
    neg_keys = ("code", "implementation", "paper", "supplementary material", "bibtex")
    if any(k in ctx for k in pos_keys):
        score += 3; reasons.append("ctx_pos")
    if any(k in ctx for k in neg_keys):
        score -= 2; reasons.append("ctx_neg")

    return score, {"reasons": reasons}


def _build_global_primary_prompt(title: str, abstract: str, cands: List[Dict], limit_chars: Optional[int] = None) -> str:
    limit = int(limit_chars or DEFAULT_LLM_CHAR_LIMIT)
    header = (
        "Task: From the candidate dataset links below, select exactly ONE link that most likely points to "
        "the dataset released BY THIS PAPER (primary dataset), not external baselines.\n"
        "Output STRICT JSON:\n"
        '{\"primary\":{\"url\":\"...\",\"reason\":\"...\",\"confidence\":0-1},\"alternatives\":[{\"url\":\"...\",\"why\":\"...\"}]}\n'
        "Decision rules:\n"
        "1) Prefer phrases like: we release / our dataset / available at / we present / data availability.\n"
        "2) Penalize: we evaluate on / we use / baseline / existing dataset.\n"
        "3) If multiple variants (mirror/resolve/file), pick the canonical landing page.\n"
        "4) The url you output MUST appear in candidates.\n"
        "5) If truly none is a released dataset, return primary.url=\"\" and explain.\n\n"
    )
    paper = f"[PAPER]\nTitle: {title}\nAbstract: {abstract}\n\n"
    used = len(header) + len(paper)
    body_lines = ["[CANDIDATES]"]

    for i, c in enumerate(cands, 1):
        # 为了控制总长，对 ctx 动态按剩余预算裁剪
        ctx = (c.get("ctx", "") or "").strip()
        # 给每项留出基本字段预算
        fixed = len(f"- #{i}\n  url: {c['url']}\n  host: {c['host']}\n  anchor: {c.get('anchor','')}\n  score_hint: {c.get('score',0)}\n  ctx: ")
        remain = max(0, limit - used - fixed - 400)  # 预留 400 结尾余量
        if remain and len(ctx) > remain:
            ctx = ctx[:remain].rsplit(" ", 1)[0] + " ..."
        line = (
            f"- #{i}\n  url: {c['url']}\n  host: {c['host']}\n"
            f"  anchor: {c.get('anchor','')}\n  score_hint: {c.get('score',0)}\n"
            f"  ctx: {ctx}\n"
        )
        body_lines.append(line)
        used += len(line)
        if used >= limit:
            break

    return header + paper + "\n".join(body_lines)


def _pick_primary_link_via_llm_global(
    rec: dict,
    links_detail: List[Dict],
    windows_meta: List[Dict],
    ds_items: List[Dict],
    topk: int = 12,
    char_limit: int = DEFAULT_LLM_CHAR_LIMIT
) -> Optional[Dict]:
    """
    让 LLM 在“所有候选数据集链接”里做唯一选择。
    返回形如：{"url":..., "desc":..., "why":..., "primary_score": float, "llm_conf": float}
    """
    from urllib.parse import urlsplit

    if not ds_items:
        return None

    # 1) 为每个候选准备压缩上下文与打分提示
    anchor_by_url = {ld.get("url",""): (ld.get("anchor") or "").strip() for ld in links_detail}
    ctx_by_url = {}
    for ld, meta in zip(links_detail, windows_meta):
        url = ld.get("url","")
        # 取 ~6 句窗口，并用 _latex_to_text 清洗
        raw_sents = (meta.get("window_sentences") or [])[:6]
        clean = [ _latex_to_text(s) for s in raw_sents ]
        ctx_by_url[url] = " ".join([s for s in clean if s])[:2000]

    # 2) 规则打分作 hint（已有 _score_candidate）
    packed = []
    seen = set()
    for it in ds_items:
        u = (it.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        nu = _norm_url(u)
        if not _is_structural_dataset_url(nu):
            continue
        host = urlsplit(u).netloc.lower()
        anchors = [anchor_by_url.get(u,"")]
        sc, _dbg = _score_candidate(nu, anchors, [ctx_by_url.get(u,"")])
        packed.append({
            "url": u,
            "host": host,
            "anchor": anchor_by_url.get(u,""),
            "ctx": ctx_by_url.get(u,""),
            "score": sc,
            "desc": (it.get("desc") or "").strip(),
            "why":  (it.get("why")  or "").strip()
        })

    if not packed:
        return None

    # 3) 选前 topk 进 prompt，避免超长
    packed.sort(key=lambda x: x["score"], reverse=True)
    cands = packed[:topk]

    # 4) 构造全局 prompt -> LLM
    prompt = _build_global_primary_prompt(
        rec.get("title",""), rec.get("abstract",""), cands, char_limit
    )
    # 正确：构造 DashScope/OpenAI 兼容的 payload，并解析 content
    system_txt = (
        "You are a link-selection assistant. Only return STRICT JSON. "
        "Follow the rules to pick exactly ONE primary dataset link from candidates."
    )
    payload = {
        "model": CONFIG.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_txt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }
    try:
        resp = _post_qwen_with_retry(payload, timeout=(10, 120), tries=3)
        content = resp["choices"][0]["message"]["content"]
        obj = _safe_json_loads(content)    
    
    
    
    except Exception as e:
        core_logger.warning("[LLM PRIMARY] parse error: %s", e)
        obj = {}

    # 5) 解析结果并校验：url 必须来自 candidates
    primary = (obj.get("primary") if isinstance(obj, dict) else None) or {}
    url = (primary.get("url") or "").strip()
    # 允许轻度归一化后比对，避免 query/末尾斜杠导致的误判
    cand_set = {_norm_url(c["url"]) for c in cands}
    if not url or _norm_url(url) not in cand_set:
        core_logger.info("[LLM PRIMARY] no valid primary chosen; skip writing.")
        return None

    # 找回原条目
    ref = next(c for c in cands if c["url"] == url)
    return {
        "url": url,
        "desc": ref.get("desc",""),
        "why":  (primary.get("reason") or ref.get("why") or ref.get("anchor") or "").strip(),
        "primary_score": float(ref["score"]),
        "llm_conf": float(primary.get("confidence", 0.0) or 0.0)
    }


def _write_primary_dataset_item(rec: dict, item: dict):
    u = (item.get("url") or "").strip()
    if not u or not _is_structural_dataset_url(_norm_url(u)):
        return

    # 仅信任 LLM 侧给出的描述；没有就放弃该条
    raw_line = (item.get("desc") or item.get("why") or "").strip()
    line = _sanitize_llm_summary_line(raw_line, max_len=None)  # 不截断

    # 若 LLM 没有给出可用描述，直接放弃（不再写入“host — anchor”或论文级描述）
    if not line:
        core_logger.info("[FINAL DESC] skip: empty LLM description for %s", u)
        return

    ts = now_iso()
    final_records = [{
        "Paper Link": rec["paper_link_pdf"],
        "Dataset Link": u,
        "Dataset Description": line,   # 纯文本，无链接/域名
        "Timestamp": ts
    }]
    core_logger.info("[FINAL DESC] %s len=%d preview=%s",
                     u, len(line), re.sub(r"\s+", " ", line)[:80])
    append_final_records(final_records)
    rec["_final_written"] = True  # ★ 标记这一篇已完成“最终写入”




def _pick_primary_link_via_llm_from_cands(rec: dict, cands: List[dict], char_limit: int):
    """对已打分的候选（url/anchor/ctx/score）做一次性 LLM 甄别。"""
    if not cands: return None
    # 给 _build_global_primary_prompt 填上 host/ctx/score
    from urllib.parse import urlsplit
    packed = []
    seen = set()
    for c in cands:
        u = (c.get("url") or "").strip()
        if not u or u in seen: continue
        seen.add(u)
        host = urlsplit(u).netloc.lower()
        packed.append({
            "url": u,
            "host": host,
            "anchor": c.get("anchor",""),
            "ctx": (c.get("ctx","") or "")[:2000],
            "score": int(c.get("score", 0)),
        })
    if not packed: return None
    prompt = _build_global_primary_prompt(rec.get("title",""), rec.get("abstract",""), packed, char_limit)
    
    try:
        payload = {
            "model": CONFIG.LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a link-selection assistant. Only return STRICT JSON."
                },
                {"role": "user", "content": prompt},
            ],
           "temperature": 0.2,
            "max_tokens": 640,
        }
        resp = _post_qwen_with_retry(payload, timeout=(10, 120), tries=3)
        content = (
            resp.get("choices", [{}])[0].get("message", {}).get("content")
            if isinstance(resp, dict) else str(resp)
        )
        obj = _safe_json_loads(content)
    except Exception as e:
        core_logger.warning("[LLM PRIMARY 1shot] parse error: %s", e)
        obj = {}
        
        
    primary = (obj.get("primary") if isinstance(obj, dict) else None) or {}
    url = (primary.get("url") or "").strip()
    if not url or url not in {c["url"] for c in packed}:
        return None
    ref = next(c for c in packed if c["url"] == url)
    return {
        "url": url,
        "desc": "",
        "why": (primary.get("reason") or ref.get("anchor","") or "").strip(),
        "primary_score": float(ref.get("score", 0)),
        "llm_conf": float(primary.get("confidence", 0.0) or 0.0),
        }
        
        
def _run_latex_stage(cat: str, results: list):
    """
    LaTeX 流程（规则优先、LLM兜底）：
      下载 -> 解压 -> 解析链接 -> 构造窗口 -> 结构化闸门 + 规则打分 直选主链
      -> 若不够强再对Top-K做一次性LLM甄别 -> 写四列表
      -> （可选）把安全摘要追加到 dataset_description
    """
    if not getattr(CONFIG, 'LATEX_ENABLED', False):
        core_logger.info("[LaTeX] disabled by CONFIG.LATEX_ENABLED")
        return

    bert_tok_name  = getattr(CONFIG, "BERT_TOKENIZER", "bert-base-uncased")
    min_sentences  = int(getattr(CONFIG, "CTX_MIN_SENTENCES", 3))
    context_size   = int(getattr(CONFIG, "CTX_CONTEXT_SIZE", 2))
    probe_window   = int(getattr(CONFIG, "CTX_PROBE_WINDOW", max(6, context_size * 3)))
    topk_primary   = int(getattr(CONFIG, "PRIMARY_TOPK", 10))
    tokenizer = AutoTokenizer.from_pretrained(bert_tok_name)
    sep_token = tokenizer.sep_token or "</s>"

    day = ledger_day_name()
    have_src = 0
    total = len(results)

    for rec in results:
        aid = rec["arxiv_id"]
        tex_root = os.path.join(CONFIG.OUT_LATEX, day, cat, aid)
        tar_path = os.path.join(tex_root, f"{aid}.tar")
        src_dir  = os.path.join(tex_root, "src")
        ensure_dir(tex_root)

        # 1) 下载源码
        mon = _start_speed_monitor(tar_path, tag=f"SRC:{aid}")        
        try:
            ok_dl = download_latex_source(aid, tar_path)
        finally:
            mon.set()
        if not ok_dl:
            core_logger.info("[LaTeX] no source / download failed: %s", aid)
            continue
        have_src += 1

        # 2) 解压
        ok_ex = extract_tar(tar_path, src_dir)
        if not ok_ex:
            core_logger.warning("[LaTeX] extract failed: %s", aid)
            continue

        # 2.5) 展开后的纯文本副本
        flat_txt_path = os.path.join(tex_root, f"{aid}.latex.txt")
        _save_latex_plaintext(src_dir, flat_txt_path)
        core_logger.info("[LaTeX] saved plaintext: %s", flat_txt_path)

        # 3) 解析链接（带前后文句子）
        links_detail = parse_links_with_sentence_context(src_dir, window=probe_window)
        if not links_detail:
            core_logger.info("[LaTeX] no links parsed: %s", aid)
            continue

        # 4) 为每个链接构造窗口（仅用于打分/摘要，不拼 description）
        windows_meta = []
        for it in links_detail:
            before = it.get("ctx_before", []) or []
            after  = it.get("ctx_after", []) or []
            anchor = it.get("anchor", "")
            url    = it["url"]
            pivot_text = f"[URL_PIVOT] {anchor or url}"
            window_sents = _build_single_window_around_pivot(
                tokenizer=tokenizer,
                sep_token=sep_token,
                before_sents=before,
                after_sents=after,
                pivot_text=pivot_text,
                max_length=10000,
                min_sentences=min_sentences,
                context_size=context_size
            ) or [pivot_text]
            windows_meta.append({"url": url, "window_sentences": window_sents})

        # 5) 结构化闸门 + 规则打分（不再做 LLM 分类）
        cands, seen_norm = [], set()
        for ld, meta in zip(links_detail, windows_meta): #这里没问题
            url = (ld.get("url") or "").strip()
            if not url: continue
            norm = _norm_url(url)
            if norm in seen_norm: continue
            anchor = (ld.get("anchor") or "").strip()
            raw_sents = (meta.get("window_sentences") or [])[:6]
            clean_ctx_parts = []
            for s in raw_sents:
                t = _latex_to_text(s)
                if t: clean_ctx_parts.append(t)
            clean_ctx = " ".join(clean_ctx_parts)[:2000]
            sc, dbg = _score_candidate_v2(url, anchor, clean_ctx)
            cands.append({"url": url, "anchor": anchor, "ctx": clean_ctx, "score": sc, "reasons": dbg.get("reasons",[])})
            seen_norm.add(norm)

        # ====== (5.5) 可选：用 LLM 做链接分类（dataset/code/other），只保留 dataset ======
        if (CONFIG.LINK_CLASSIFIER == "llm" and not getattr(CONFIG, "LATEX_LLM_DISABLED", False)):
            qwen_inputs = []
            for c in cands:
                qwen_inputs.append({
                    "url": c["url"],
                    "input": _truncate_for_llm(
                        f"[URL] {c['url']}\n[ANCHOR] {c.get('anchor','')}\n[CONTEXT] {c.get('ctx','')}",
                        DEFAULT_LLM_CHAR_LIMIT
                    )
                })
            items = classify_link_inputs_via_llm_safe(
                qwen_inputs, batch_size=4, max_input_length=DEFAULT_LLM_CHAR_LIMIT
            )
            keep = {it.get("url","") for it in items if (it.get("label") == "dataset")}
            before_n = len(cands)
            cands = [c for c in cands if c["url"] in keep]
            core_logger.info("[LaTeX][cls-llm] %s filtered dataset-like: %d -> %d",
                             aid, before_n, len(cands))

            if not cands:
                core_logger.info("[LaTeX] %s no dataset links after LLM classification.", aid)
                continue 

        # ====== (6) 主链甄别：支持 rule / hybrid / llm 三种模式 ======
        picker_mode = (getattr(CONFIG, "LINK_PICKER", "hybrid") or "hybrid").lower()
        if getattr(CONFIG, "LATEX_LLM_DISABLED", False):
            picker_mode = "rule"  # 全局禁用时强制走规则

        # 6.1 始终先算一次“规则候选”（供 rule/hybrid 使用）
        preferred = _pick_by_preferred_hosts(cands, topk=getattr(CONFIG, "PRIMARY_HOST_TOPK", 5))
        if preferred and not getattr(CONFIG, "DISABLE_PREFERRED_PICK", False):
            picked_rule = preferred
        else:
            picked_rule = _choose_primary_rule_only(cands)

        def _write_pick(tag, item):
            _write_primary_dataset_item(rec, item)
            core_logger.info("[LaTeX][%s] %s primary: %s (score=%.2f conf=%.2f)",
                             tag, aid, item["url"], item.get("primary_score", 0.0), item.get("llm_conf", 0.0))

        def _fallback_top1():
            if not cands: 
                return None
            top1 = max(cands, key=lambda x: x.get("score", 0))  # 注意：别用未排序的 cands[0]
            return {
                "url": top1["url"], "desc": "", "why": top1.get("anchor",""),
                "primary_score": float(top1.get("score", 0)), "llm_conf": 0.0
            }

        if picker_mode == "rule":
            if picked_rule:
                _write_pick("rule", {
                    "url": picked_rule["url"], "desc": "", "why": picked_rule.get("why",""),
                    "primary_score": picked_rule["primary_score"], "llm_conf": 0.0
                })
            elif getattr(CONFIG, "FALLBACK_TOP1_ALWAYS", True):
                fb = _fallback_top1()
                if fb: _write_pick("fallback-top1", fb)

        elif picker_mode == "llm":
            llm_pick = _pick_primary_link_via_llm_from_cands(
                rec=rec,
                cands=sorted(cands, key=lambda x: x.get("score", 0), reverse=True)[:topk_primary],
                char_limit=DEFAULT_LLM_CHAR_LIMIT
            )
            if llm_pick:
                _write_pick("llm", llm_pick)
            elif getattr(CONFIG, "FALLBACK_TOP1_ALWAYS", True):
                fb = _fallback_top1()
                if fb: _write_pick("fallback-top1", fb)

        else:  # "hybrid"（默认）：规则优先，规则不稳/未选出再让 LLM 选
            if picked_rule:
                _write_pick("rule", {
                    "url": picked_rule["url"], "desc": "", "why": picked_rule.get("why",""),
                    "primary_score": picked_rule["primary_score"], "llm_conf": 0.0
                })
            else:
                llm_pick = _pick_primary_link_via_llm_from_cands(
                    rec=rec,
                    cands=sorted(cands, key=lambda x: x.get("score", 0), reverse=True)[:topk_primary],
                    char_limit=DEFAULT_LLM_CHAR_LIMIT
                )
                if llm_pick:
                    _write_pick("llm", llm_pick)
                elif getattr(CONFIG, "FALLBACK_TOP1_ALWAYS", True):
                    fb = _fallback_top1()
                    if fb: _write_pick("fallback-top1", fb)
                    
                    
                    
        # 7) （可选）把安全摘要（域名 + 锚文本清洗）追加进 dataset_description
        if getattr(CONFIG, "APPEND_LINK_SUMMARIES_TO_DESC", True) and cands:
            from urllib.parse import urlsplit
            safe_lines, seen_line = [], set()
            for c in sorted(cands, key=lambda x: x["score"], reverse=True)[:8]:
                dom  = urlsplit(c["url"]).netloc.lower()
                raw  = c.get("anchor","") or "dataset link"
                line = _sanitize_llm_summary_line(raw) or "dataset link"
                pretty = f"{dom} — {line}"
                if pretty not in seen_line:
                    seen_line.add(pretty); safe_lines.append(pretty)
            if safe_lines:
                block = "[DATASET LINKS]\n" + "\n".join(f"- {s}" for s in safe_lines)
                proc_path = os.path.join(CONFIG.OUT_PROCESSED, day, cat, f"{aid}.partial.json")
                try:
                    ensure_dir(os.path.dirname(proc_path))
                    doc = {}
                    if os.path.exists(proc_path):
                        with open(proc_path, "r", encoding="utf-8") as f:
                            doc = json.load(f) or {}
                    base = (doc.get("dataset_description") or rec.get("dataset_description") or "").strip()
                    if block not in base:
                        new_desc = (base + ("\n\n" if base else "") + block) if base else block
                        doc["dataset_description"] = new_desc
                        rec["dataset_description"] = new_desc
                        with open(proc_path, "w", encoding="utf-8") as f:
                            json.dump(doc, f, ensure_ascii=False, indent=2)
                        core_logger.info("[LaTeX] description appended (SAFE rule summaries) for %s", aid)
                    else:
                        rec["dataset_description"] = base
                except Exception as e:
                    core_logger.warning("[LaTeX] partial.json append failed: %s", e)

        # 8) 审计 sidecar：记录候选分数/原因
        side = {
            "arxiv_id": aid,
            "paper_link_pdf": rec["paper_link_pdf"],
            "candidates": [
                {"url": c["url"], "anchor": c.get("anchor",""), "score": c.get("score",0), "reasons": c.get("reasons", [])}
                for c in sorted(cands, key=lambda x: x["score"], reverse=True)
            ],
        }
        side_path = os.path.join(tex_root, f"{aid}.links.json")
        try:
            with open(side_path, "w", encoding="utf-8") as f:
                json.dump(side, f, ensure_ascii=False, indent=2)
        except Exception as e:
            core_logger.warning("[LaTeX] write side json failed: %s", e)

        core_logger.info(
            "[LaTeX] %s: parsed=%d, structural=%d (rule-picked: %s)",
            aid, len(links_detail), len(cands),
            "yes" if picked_rule else "no"
        )


    ratio = (have_src / total * 100.0) if total else 0.0
    core_logger.info("[LaTeX] %s coverage: %d / %d (%.1f%%)", cat, have_src, total, ratio)


def _save_latex_plaintext(src_dir: str, out_txt_path: str) -> None:
    """把 LaTeX 源目录下的文本文件合并到一个 .txt 里，方便检索/审计。"""
    exts = ('.tex', '.bib', '.md', '.txt', '.sty', '.cls')
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(src_dir):
            for fn in sorted(files):
                if not fn.lower().endswith(exts):
                    continue
                fp = os.path.join(root, fn)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        rel = os.path.relpath(fp, src_dir)
                        out.write(f"\n\n===== {rel} =====\n")
                        out.write(f.read())
                except Exception as e:
                    core_logger.warning("[LaTeX] skip non-text or unreadable file %s: %s", fp, e)


# ========================================================================
# =                              数据集落盘                               =
# ========================================================================

def _append_jsonl(path: str, obj: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _dump_link_dataset_samples(
    cat: str,
    aid: str,
    paper_link_pdf: str,
    links_detail: List[Dict],
    qwen_inputs: List[Dict],
    items: List[Dict],
    windows_meta: List[Dict],
) -> str:
    """
    将“URL+上下文 -> LLM标签”的样本追加写入 JSONL。
    路径: <CONFIG.OUT_LATEX>/dataset/<day>/<cat>.jsonl
    每行包含：url/label/why/上下文/窗口句子/文件位置信息/模型超参 等。
    """
    day = ledger_day_name()
    out_root = os.path.join(CONFIG.OUT_LATEX, "dataset")
    out_dir  = os.path.join(out_root, day)
    out_path = os.path.join(out_dir, f"{cat}.jsonl")
    ts = now_iso()

    for ld, qi, lab, meta in zip(links_detail, qwen_inputs, items, windows_meta):
        row = {
            "arxiv_id": aid,
            "category": cat,
            "paper_link_pdf": paper_link_pdf,
            "url": ld.get("url", ""),
            "label": lab.get("label", "other"),
            "why": lab.get("why", ""),
            "anchor": ld.get("anchor", ""),
            "section": ld.get("section", ""),
            "file": ld.get("file", ""),
            "line": ld.get("line", -1),
            "ctx_before": ld.get("ctx_before", []),
            "ctx_after": ld.get("ctx_after", []),
            "window_sentences": meta.get("window_sentences", []),
            "input": qi.get("input", ""),
            # 记录窗口构造的关键超参，便于复现实验
            "sep_token": meta.get("sep_token", ""),
            "max_length": meta.get("max_length", 0),
            "min_sentences": meta.get("min_sentences", 0),
            "context_size": meta.get("context_size", 0),
            "timestamp": ts,
        }
        _append_jsonl(out_path, row)

    core_logger.info("[DATASET] appended %d labeled link sample(s) -> %s", len(items), out_path)
    return out_path



def append_link_corpus(records: List[Dict]):
    """
    记录每条链接的分类，用于构建“分类数据集”。
    字段示例：
    {
      "Arxiv ID": "2509.01234v1",
      "Category": "cs.CL",
      "Paper Link": "https://arxiv.org/pdf/2509.01234v1.pdf",
      "URL": "https://github.com/xxx/yyy",
      "Label": "dataset|code|other",
      "Why": "LLM 的分类理由（可空）",
      "Timestamp": "2025-10-16T12:34:56+08:00"
    }
    """
    day = ledger_day_name()
    out_dir = CONFIG.OUT_FINAL
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{day}.links.json")

    with _FINAL_WRITE_LOCK:
        existing = []
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        merged = existing + (records or [])
        seen, dedup = set(), []
        for r in merged:
            key = (_norm_url(r.get("Paper Link","")), _norm_url(r.get("URL","")), r.get("Label",""))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dedup, f, ensure_ascii=False, indent=2)
    core_logger.info("Link corpus written: %s (+%d)", out_path, len(records))


# ---- 超低侵入性能计时：统一包裹常用慢函数 ----
from time import time as _now

# 可按需调阈值（毫秒）
SLOW = {
    "pdf_download_ms":          120_000,
    "grobid_ms":                 60_000,
    "txt_to_json_ms":            15_000,
    "bert_infer_ms":             20_000,
    "latex_src_download_ms":     20_000,
    "latex_extract_ms":           8_000,
    "latex_parse_links_ms":      12_000,
    "latex_build_windows_ms":     6_000,   
    "latex_score_filter_ms":      4_000,
    "latex_pick_primary_ms":      3_000,   #
}

_AID_RE = re.compile(r'(\d{4}\.\d{5}(?:v\d+)?)', re.I)

def _infer_aid_from_vals(vals):
    for v in vals:
        try:
            s = str(v)
        except Exception:
            continue
        m = _AID_RE.search(s)
        if m:
            return m.group(1)
    return "?"

def _wrap_timed(func, name: str, warn_key: str = None):
    def _wrapped(*args, **kwargs):
        t0 = _now()
        try:
            return func(*args, **kwargs)
        finally:
            ms = int((_now() - t0) * 1000)
            aid = _infer_aid_from_vals(list(args) + list(kwargs.values()))
            thr = SLOW.get(warn_key or "", 10**9)
            lvl = logging.WARNING if ms > thr else logging.INFO
            try:
                core_logger.log(lvl, "[TIME][%s] %s = %d ms", aid, name, ms)
            except Exception:
                # 避免日志异常影响主流程
                pass
    return _wrapped

def install_perf_probes():
    """把关键函数统一套上计时包装；一次调用，全局生效。"""
    global download_pdf, grobid_extract_to_txt, txt_to_json
    global download_latex_source, extract_tar, parse_links_with_sentence_context
    global _post_qwen_with_retry
    # —— PDF→TXT→JSON
    download_pdf              = _wrap_timed(download_pdf,              "pdf_download",         "pdf_download_ms")
    grobid_extract_to_txt     = _wrap_timed(grobid_extract_to_txt,     "grobid",               "grobid_ms")
    txt_to_json               = _wrap_timed(txt_to_json,               "txt_to_json",          "txt_to_json_ms")
    # —— LaTeX 源处理
    download_latex_source     = _wrap_timed(download_latex_source,     "latex_src_download",   "latex_src_download_ms")
    extract_tar               = _wrap_timed(extract_tar,               "latex_extract",        "latex_extract_ms")
    parse_links_with_sentence_context = _wrap_timed(
        parse_links_with_sentence_context, "latex_parse_links", "latex_parse_links_ms"
    )
    # —— LLM（用于 1-shot 甄别）
    _post_qwen_with_retry     = _wrap_timed(_post_qwen_with_retry,     "llm_call",             "latex_pick_primary_ms")
    # 如需观察 BERT 整体耗时，也可以（可选）把 label_sentences_via_bert 包起来：
    # global label_sentences_via_bert
    # label_sentences_via_bert  = _wrap_timed(label_sentences_via_bert, "bert_label",           "bert_infer_ms")



# ========================================================================
# =                               主循环                                 =
# ========================================================================

def main_loop(stop_event=None):
    # 初始化轻量状态
    ensure_dir(CONFIG.STATE_DIR)
    state = JsonState(os.path.join(CONFIG.STATE_DIR, "arxiv_state.json"))
    state._data.setdefault("seen_ids", {})
    state._data.setdefault("last_updated", {})
    state.save()

    ensure_dir(CONFIG.OUT_PAPERS)

    bert = BertGate()

    while True:
        # 若收到 stop 信号，退出循环
        if stop_event is not None and stop_event.is_set():
            core_logger.info("Stop event received, exit main_loop.")
            break

        start_loop = time.time()

        processed_ids: set = set()

        def _seen_all_snapshot() -> set:
            all_ids = set()
            for _c in CONFIG.CATEGORIES:
                try:
                    all_ids |= set(state.seen(_c))
                except Exception:
                    pass
            return all_ids

        seen_all = _seen_all_snapshot()

        for cat in CONFIG.CATEGORIES:
            if stop_event is not None and stop_event.is_set():
                core_logger.info("Stop event received, break category loop.")
                break

            try:
                time.sleep(random.uniform(*CONFIG.JITTER_SEC))
                core_logger.info("Polling %s ...", cat)
                # 下面保持你的原逻辑不动……
                # fetch_arxiv_recent / BERT 过滤 / process_until_description / _run_latex_stage / state.save()
                time.sleep(random.uniform(*CONFIG.JITTER_SEC))
                core_logger.info("Polling %s ...", cat)

                # ============= 1 爬取 =============
                entries = []
                start = 0
                page_size = CONFIG.ARXIV_MAX_RESULTS
                while True:
                    page = fetch_arxiv_recent(cat, max_results=page_size, start=start)
                    if not page:
                        break
                    entries.extend(page)
                    if len(page) < page_size:
                        break
                    start += page_size

                core_logger.info("Fetched %s entries in %s (LAST_N_DAYS=%s)", len(entries), cat, CONFIG.LAST_N_DAYS)

                # ============= 2 全局过滤（历史 seen + 本轮 processed） =============
                candidates = [e for e in entries
                              if e["id"] not in seen_all and e["id"] not in processed_ids]
                core_logger.info("Candidates after global filter: %s (seen_all=%s, processed=%s)",
                                 len(candidates), len(seen_all), len(processed_ids))

                # ============= 3 Bert过滤 =============
                prefiltered = []
                for e in candidates:
                    prob = bert.score(e["title"], e["abstract"])
                    e["_bert_prob"] = float(prob)
                    if prob >= CONFIG.BERT_THRESHOLD:
                        prefiltered.append(e)
                    else:
                        core_logger.info("[%s] %s filtered by BERT (%.2f)", cat, e["id"], prob)

                if not prefiltered:
                    core_logger.info("No entries to process for %s", cat)
                    continue

                # ============= 4 dataset_description =============
                results = []
                with ThreadPoolExecutor(max_workers=4) as ex:
                    fut2entry = {ex.submit(process_until_description, cat, e): e for e in prefiltered}
                    for fut in as_completed(fut2entry):
                        e = fut2entry[fut]
                        try:
                            rec = fut.result()
                            results.append(rec)

                            # —— 成功后：标记为全局已处理 —— 
                            processed_ids.add(e["id"])
                            # 写入所有 cat 的 seen，确保后续分类/下轮也不会再处理
                            for _c in CONFIG.CATEGORIES:
                                state.add_seen(_c, e["id"])
                            state.set_last_updated(cat, e["updated"])
                            state.save()
                            # 更新本轮的全局快照，后面的分类能立即看到
                            seen_all.add(e["id"])

                        except Exception as err:
                            core_logger.exception("Process failed on %s/%s: %s", cat, e.get("id"), err)

                core_logger.info("Processed %d items for %s (until description stage).", len(results), cat)

                # ============= 5 Latex获取 =============
                accepted_for_latex = [r for r in results if r.get("desc_gate", {}).get("accepted", False)]
                _run_latex_stage(cat, accepted_for_latex)
            except Exception as e:
                core_logger.exception("Loop error on %s: %s", cat, e)

        # 轮询休眠，期间也要响应 stop_event
        elapsed = time.time() - start_loop
        sleep_sec = max(0, CONFIG.POLL_MINUTES * 60 - elapsed)
        core_logger.info("Cycle done (took %.1fs), sleep %.1fs", elapsed, sleep_sec)

        end_time = time.time() + sleep_sec
        while time.time() < end_time:
            if stop_event is not None and stop_event.is_set():
                core_logger.info("Stop event received during sleep, exit main_loop.")
                return
            time.sleep(1.0)





if __name__ == "__main__":
    CONFIG.LAST_N_DAYS = 150
    setup_logging()
    install_perf_probes()
    core_logger.info("Starting realtime loop (stop before link extraction)...")
    main_loop()  # 直接跑时还是无限循环