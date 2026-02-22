#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合流程：
A) 先合并 final_dataset 目录下所有“非 final_dataset.json”的 JSON 分片；
B) 再遍历 2_paper_processed/<date>/<category>/*.partial.json 生成新条目（新增）；
C) 再遍历 3_pdf/<date>/<category> 生成新条目；
D) 与现有 final_dataset.json 合并：按 arXiv ID 去重，只填补空字段，保留键序与写出风格；
E) 写出前剔除 Dataset Description 为空的记录（可通过开关控制）。
"""
import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from collections import OrderedDict

# ========= 根目录与输出路径 =========
ROOT_2PROCESSED = Path("/home/kzlab/muse/Savvy/Data_collection/output/2_paper_processed")  # 新增
ROOT_3PDF = Path("/home/kzlab/muse/Savvy/Data_collection/output/3_pdf")
FINAL_DIR = Path("/home/kzlab/muse/Savvy/Data_collection/script/retrival/dataset_retrieval/dataset_retrieval/datasets")
OUTPUT_PATH = FINAL_DIR / "final_dataset.json"
# 输出风格: "pretty"（多行，默认） | "compact" | "preserve"（沿用旧文件风格）
STYLE_MODE = "pretty"
PRETTY_INDENT = 2
# 是否在最终写出前删除 “Dataset Description” 为空的记录
PRUNE_EMPTY_DESCRIPTION = True
# ==================================

URL_RE = re.compile(r'https?://[^\s<>\]\)"}]+', re.IGNORECASE)
DATASET_DOMAINS = [
    "huggingface.co/datasets","kaggle.com/datasets","zenodo.org/record","zenodo.org",
    "figshare.com","osf.io","data.mendeley.com","paperswithcode.com/datasets",
    "datahub.io","doi.org","storage.googleapis.com","drive.google.com",
    "cloud.google.com/storage","github.com",
]
ID_PATTERN = re.compile(r'^(\d{4}\.\d+v\d+)')          # 文件名前缀 e.g., 2401.06915v3
ARXIV_URL_ID_RE = re.compile(r'/pdf/(\d{4}\.\d+v\d+)\.pdf$', re.IGNORECASE)
PARTIAL_NAME_RE = re.compile(r'^(\d{4}\.\d+v\d+)\.partial\.json$')  # 新增
FOUR_FIELDS = ["Paper Link","Dataset Link","Dataset Description","Timestamp"]

# --------------------- 工具函数 ---------------------
def extract_id(name: str):
    m = ID_PATTERN.match(name)
    return m.group(1) if m else None

def get_id_from_paper_link(paper_link: str):
    if not isinstance(paper_link, str):
        return None
    m = ARXIV_URL_ID_RE.search(paper_link.strip())
    return m.group(1) if m else None

def is_blank(v):
    return (v is None) or (isinstance(v, str) and v.strip() == "")

def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def parse_json_texts(json_path: Path):
    texts = []
    if not json_path.exists():
        return texts
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return texts
    data = obj.get("data", []) if isinstance(obj, dict) else []
    for item in data:
        ft = item.get("full_text", "")
        if isinstance(ft, str):
            texts.append(ft)
        for s in item.get("sentences", []) or []:
            t = s.get("text", "")
            if isinstance(t, str):
                texts.append(t)
    return texts

def best_dataset_link(corpus_texts):
    urls = []
    for t in corpus_texts:
        for u in URL_RE.findall(t or ""):
            urls.append(u.rstrip('.,;:)]}"\''))
    urls = [u for u in urls if "arxiv.org" not in u.lower()]
    if not urls:
        return ""
    def score(u: str):
        lu = u.lower()
        for rank, dom in enumerate(DATASET_DOMAINS):
            if dom in lu:
                return (rank, len(u))
        return (999, len(u))
    urls.sort(key=score)
    return urls[0]

def file_timestamp(paths: Iterable[Path]):
    for p in paths:
        if p and p.exists():
            try:
                ts = p.stat().st_mtime
                dt = _dt.datetime.fromtimestamp(ts).astimezone()
                return dt.isoformat(timespec="seconds")
            except Exception:
                continue
    return ""  # 占位

def detect_style(raw_text: str):
    """返回 (indent, compact)。若文本为空，默认多行(indent=2, compact=False)。"""
    if not raw_text.strip():
        return PRETTY_INDENT, False
    compact = ("\n" not in raw_text.strip())
    if compact:
        return 0, True
    indent_guess = PRETTY_INDENT
    for line in raw_text.splitlines():
        if line.lstrip() != line and ":" in line:
            spaces = len(line) - len(line.lstrip(" "))
            if spaces > 0:
                indent_guess = spaces
                break
    return indent_guess, False

# --------------------- 规范化 ---------------------
def ensure_placeholders_in_place(od: OrderedDict):
    """确保四字段存在；None→空串；缺失键追加到末尾。"""
    for k in FOUR_FIELDS:
        if k in od:
            v = od[k]
            if v is None:
                od[k] = ""
            elif not isinstance(v, str):
                od[k] = str(v)
        else:
            od[k] = ""

def normalize_entry_keys(od: OrderedDict) -> OrderedDict:
    """
    修复异常键：若存在空键 "" 且值像 arXiv PDF，则视为 Paper Link；
    按 FOUR_FIELDS 顺序放前，其余键原序保留。
    """
    if "" in od and "Paper Link" not in od:
        val = od.get("", "")
        if isinstance(val, str) and "arxiv.org/pdf/" in val:
            od["Paper Link"] = val
        del od[""]
    ensure_placeholders_in_place(od)

    reordered = OrderedDict()
    for k in FOUR_FIELDS:
        reordered[k] = od.get(k, "")
    for k, v in od.items():
        if k not in reordered:
            reordered[k] = v
    return reordered

# --------------------- 构造条目（从 3_pdf ） ---------------------
def build_entries_from_category_dir(cat_dir: Path) -> List[OrderedDict]:
    ids = {extract_id(n) for n in os.listdir(cat_dir) if extract_id(n)}
    entries: List[OrderedDict] = []
    for _id in sorted(ids):
        paths = {
            "json": cat_dir / f"{_id}.json",
            "pdf":  cat_dir / f"{_id}.pdf",
            "desc": cat_dir / f"{_id}.bert.desc.txt",
            "txt":  cat_dir / f"{_id}.txt",
        }
        paper_link = f"https://arxiv.org/pdf/{_id}.pdf"
        dataset_desc = safe_read(paths["desc"]).strip()

        corpus_texts = []
        corpus_texts.extend(parse_json_texts(paths["json"]))
        if dataset_desc:
            corpus_texts.append(dataset_desc)
        txt_content = safe_read(paths["txt"]).strip()
        if txt_content:
            corpus_texts.append(txt_content)

        dataset_link = best_dataset_link(corpus_texts) if corpus_texts else ""
        timestamp = file_timestamp([paths["json"], paths["pdf"], paths["desc"], paths["txt"]])

        od = OrderedDict()
        od["Paper Link"] = paper_link
        od["Dataset Link"] = dataset_link if isinstance(dataset_link, str) else ""
        od["Dataset Description"] = dataset_desc if isinstance(dataset_desc, str) else ""
        od["Timestamp"] = timestamp if isinstance(timestamp, str) else ""
        entries.append(od)
    return entries

def find_category_dirs(root_dir: Path) -> List[Path]:
    result: List[Path] = []
    if not root_dir.exists():
        return result
    for date_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        for cat_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
            has_id_prefix = any(extract_id(n) for n in os.listdir(cat_dir) if not n.startswith("."))
            if has_id_prefix:
                result.append(cat_dir)
    return result

# --------------------- 读取 final_dataset 目录中的分片 ---------------------
def parse_as_entry_list(obj) -> List[OrderedDict]:
    """将各种可能结构解析为 [OrderedDict, ...]，只保留含 Paper Link 的条目。"""
    records: List[OrderedDict] = []
    if isinstance(obj, list):
        itr = obj
    elif isinstance(obj, dict):
        # 1) 直接就是一条记录
        if any(k in obj for k in FOUR_FIELDS):
            itr = [obj]
        # 2) 某些导出结构可能包在 { "data": [...] }
        elif "data" in obj and isinstance(obj["data"], list):
            itr = obj["data"]
        else:
            itr = []
    else:
        itr = []
    for r in itr:
        if isinstance(r, dict):
            od = OrderedDict(r)
            od = normalize_entry_keys(od)
            if get_id_from_paper_link(od.get("Paper Link", "")):
                records.append(od)
    return records

def load_extra_final_jsons(final_dir: Path, exclude_path: Path) -> List[OrderedDict]:
    """读取 final_dataset 目录下所有非 exclude_path 的 .json，聚合为记录列表。"""
    out: List[OrderedDict] = []
    if not final_dir.exists():
        return out
    for p in sorted(final_dir.glob("*.json")):
        if p.resolve() == exclude_path.resolve():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            obj = json.loads(txt, object_pairs_hook=OrderedDict)
            out.extend(parse_as_entry_list(obj))
        except Exception:
            # 忽略坏文件
            continue
    return out

# --------------------- 新增：从 2_paper_processed 构造条目 ---------------------
def iter_partial_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for date_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for cat_dir in sorted([p for p in date_dir.iterdir() if p.is_dir()]):
            for f in sorted(cat_dir.glob("*.partial.json")):
                if PARTIAL_NAME_RE.match(f.name):
                    yield f

def _pick_timestamp_from_obj_or_file(obj: dict, file_path: Path) -> str:
    # 优先 json 内的 updated/published，其次文件 mtime
    for k in ("updated", "published"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return file_timestamp([file_path]) or ""

def build_entry_from_partial_json(p: Path) -> Optional[OrderedDict]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    arxiv_id = (obj.get("arxiv_id") or "") if isinstance(obj, dict) else ""
    m = PARTIAL_NAME_RE.match(p.name)
    if not arxiv_id and m:
        arxiv_id = m.group(1)

    paper_link = ""
    if isinstance(obj, dict):
        paper_link = (obj.get("paper_link_pdf") or "").strip()
    if not paper_link and arxiv_id:
        paper_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # 重点：必须有描述；优先 dataset_description，其次 abstract 兜底（避免被 PRUNE 删掉）
    dataset_desc = ""
    if isinstance(obj, dict):
        dataset_desc = (obj.get("dataset_description") or "").strip()
        if not dataset_desc:
            dataset_desc = (obj.get("abstract") or "").strip()

    # 从描述/摘要中尝试提取一个数据集链接（可为空）
    corpus = []
    if isinstance(obj, dict):
        if isinstance(obj.get("dataset_description"), str):
            corpus.append(obj["dataset_description"])
        if isinstance(obj.get("abstract"), str):
            corpus.append(obj["abstract"])
    dataset_link = best_dataset_link(corpus) if corpus else ""

    timestamp = _pick_timestamp_from_obj_or_file(obj if isinstance(obj, dict) else {}, p)

    od = OrderedDict()
    od["Paper Link"] = paper_link
    od["Dataset Link"] = dataset_link if isinstance(dataset_link, str) else ""
    od["Dataset Description"] = dataset_desc if isinstance(dataset_desc, str) else ""
    od["Timestamp"] = timestamp if isinstance(timestamp, str) else ""
    return od

def build_entries_from_2_processed(root: Path) -> List[OrderedDict]:
    entries: List[OrderedDict] = []
    for f in iter_partial_json_files(root):
        od = build_entry_from_partial_json(f)
        if od is None:
            continue
        # 如果描述仍为空，让后续 PRUNE 过滤；否则收集
        entries.append(od)
    return entries

# --------------------- 合并策略 ---------------------
def merge_fill(existing_list: List[OrderedDict], new_entries: List[OrderedDict]) -> List[OrderedDict]:
    index: Dict[str, int] = {}
    deduped: List[OrderedDict] = []
    for e in existing_list:
        e = OrderedDict(e) if not isinstance(e, OrderedDict) else e
        e = normalize_entry_keys(e)
        k = get_id_from_paper_link(e.get("Paper Link", ""))
        if not k:
            deduped.append(e)
            continue
        if k in index:
            continue
        index[k] = len(deduped)
        deduped.append(e)

    for ne in new_entries:
        ne = normalize_entry_keys(ne)
        k = get_id_from_paper_link(ne.get("Paper Link", ""))
        if not k:
            continue
        if k in index:
            ex = deduped[index[k]]
            for fld in FOUR_FIELDS:
                if fld not in ex or is_blank(ex.get(fld, "")):
                    ex[fld] = ne.get(fld, "") if isinstance(ne.get(fld, ""), str) else ""
            deduped[index[k]] = normalize_entry_keys(ex)
        else:
            deduped.append(ne)
            index[k] = len(deduped) - 1
    return deduped

# --------------------- 过滤（剔除空描述） ---------------------
def prune_empty_description(records: List[OrderedDict]) -> List[OrderedDict]:
    """去除 Dataset Description 为空或仅空白的记录。"""
    out = []
    for r in records:
        desc = r.get("Dataset Description", "")
        if isinstance(desc, str) and desc.strip() != "":
            out.append(r)
    return out

# --------------------- 主流程 ---------------------
def run():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 读取旧的 final_dataset.json
    raw_text = OUTPUT_PATH.read_text(encoding="utf-8", errors="ignore") if OUTPUT_PATH.exists() else ""
    indent_detected, compact_detected = detect_style(raw_text)

    existing: List[OrderedDict] = []
    if raw_text.strip():
        try:
            parsed = json.loads(raw_text, object_pairs_hook=OrderedDict)
            if isinstance(parsed, list):
                existing = parsed
            elif isinstance(parsed, dict):
                existing = [parsed]
        except Exception:
            existing = []

    # 2) 先并入目录里的所有分片（不是 final_dataset.json 的 .json）
    extra_entries = load_extra_final_jsons(FINAL_DIR, OUTPUT_PATH)

    # 3) 新增：扫描 2_paper_processed
    processed_entries = build_entries_from_2_processed(ROOT_2PROCESSED)

    # 4) 再扫描 3_pdf/<date>/<category>
    cat_dirs = find_category_dirs(ROOT_3PDF)
    pdf_entries: List[OrderedDict] = []
    for d in cat_dirs:
        pdf_entries.extend(build_entries_from_category_dir(d))

    # 5) 合并：existing + extra + processed + pdf
    merged = merge_fill(existing, extra_entries + processed_entries + pdf_entries)

    # 5.1) 去除 Dataset Description 为空的条目（可配置）
    if PRUNE_EMPTY_DESCRIPTION:
        merged = prune_empty_description(merged)

    # 6) 写回，按风格输出
    if STYLE_MODE == "compact":
        text = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
    elif STYLE_MODE == "preserve":
        if compact_detected:
            text = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
        else:
            text = json.dumps(merged, ensure_ascii=False, indent=indent_detected or PRETTY_INDENT)
    else:  # pretty
        text = json.dumps(merged, ensure_ascii=False, indent=PRETTY_INDENT)

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    # 打印概要
    print(f"[OK] extra json files merged: {len([p for p in FINAL_DIR.glob('*.json') if p.name != OUTPUT_PATH.name])}")
    print(f"[OK] scanned partial jsons: {len(list(iter_partial_json_files(ROOT_2PROCESSED)))} under {ROOT_2PROCESSED}")
    print(f"[OK] scanned category dirs: {len(cat_dirs)} under {ROOT_3PDF}")
    mode_text = STYLE_MODE if STYLE_MODE != "preserve" else ("compact" if compact_detected else f"indent={indent_detected or PRETTY_INDENT}")
    print(f"[OK] wrote {len(merged)} records -> {OUTPUT_PATH} (style={mode_text})")

if __name__ == "__main__":
    run()
