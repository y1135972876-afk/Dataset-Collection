# utils/utils_bertlabel_latex_extract.py
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import random
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from torch import nn
from transformers import AutoModel
from typing import Optional, List, Dict, Tuple


#==================参数==================
from utils.utils_loop import CONFIG




#=====================工具函数======================
from utils.utils_loop import (
    logger as core_logger,
    ensure_dir, day_str, ledger_day_name, now_iso,
    JsonState, fetch_arxiv_recent, BertGate,
    download_pdf, grobid_extract_to_txt, txt_to_json,
    collect_dataset_description,
    pdf_link_from_id, append_final_records,
    # 新增：LaTeX 工具
    download_latex_source, extract_tar, parse_links_from_tex_dir, classify_links_via_llm,
    parse_links_with_sentence_context, classify_link_inputs_via_llm,
    _FINAL_WRITE_LOCK,
    _count_tokens, _build_single_window_around_pivot
)

logger = logging.getLogger("realtime")

# 保持你的原始路径不变（不做额外修改）
model_dir="/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/dataset_research/针对dataset_resize重新划分前后的研究_bert模型/bert-base-cased__train_val_revise.json__20251014_003940__exp1使用重新划分的数据集/best_model.pth"
CONFIG.BERT_TOKENIZER="/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"


def _get_window_size(tokenizer, sentences: List[Dict], start: int,
                     max_length: int, min_sentences: int) -> int:
    """与训练相同：按 token 长度从 start 起尽量扩充，至少 min_sentences。"""
    cur_tokens, win = 0, 0
    for i in range(start, len(sentences)):
        tokens = tokenizer.encode(sentences[i]['text'], add_special_tokens=False)
        next_len = cur_tokens + len(tokens) + (1 if cur_tokens > 0 else 2)  # 中间 sep + 首尾 special
        if next_len >= max_length - 2:
            break
        cur_tokens = next_len
        win += 1
    return max(min_sentences, win if win > 0 else 0)

def _build_windows(sentences: List[Dict], tokenizer, max_length: int,
                   min_sentences: int = 3, context_size: int = 2) -> List[Dict]:
    """
    复刻训练集构造：token 限制下取窗口；若窗口内含正样本，则收缩到其附近；步长按是否含正样本区分。
    推理阶段通常无正样本提示，该分支多半不触发，但保持一致性更稳。
    """
    examples = []
    i = 0
    while i < len(sentences):
        window_size = _get_window_size(tokenizer, sentences, i, max_length, min_sentences)
        end_idx = min(i + window_size, len(sentences))
        current = sentences[i:end_idx]
        indices = list(range(i, end_idx))

        has_positive = any(int(s.get("label", 0)) == 1 for s in current)
        if has_positive:
            first_pos = next(k for k, s in enumerate(current) if int(s.get("label", 0)) == 1)
            st = max(0, first_pos - context_size)
            ed = min(len(current), first_pos + context_size + 1)
            current = current[st:ed]
            indices = indices[st:ed]

        if len(current) >= min_sentences:
            examples.append({"sentences": current, "indices": indices})

        step_size = max(1, window_size // 3) if has_positive else max(1, window_size // 2)
        i += step_size
    return examples


class SentenceClassifier(nn.Module):
    """与训练时一致：backbone 输出的 token 序列经 sep 切分成句子，做句级分类"""
    def __init__(self, config, backbone: nn.Module, tokenizer, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.tokenizer = tokenizer
        hidden = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden, num_labels),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=getattr(config, 'initializer_range', 0.02))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        # outputs[0]: [B, L, H]
        seq = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
        B, L, H = seq.shape
        sep_id = self.tokenizer.sep_token_id
        device = input_ids.device
        logits_list = []

        for i in range(B):
            sep_pos = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
            # 把 CLS(0) + 每个 SEP 作为边界
            bounds = torch.cat([torch.tensor([0], device=device), sep_pos])
            sent_reps = []
            for j in range(len(bounds) - 1):
                st, ed = bounds[j], bounds[j+1]
                tok = seq[i, st:ed+1]                                 # [l,H]
                msk = attention_mask[i, st:ed+1].unsqueeze(-1)        # [l,1]
                rep = (tok * msk).sum(dim=0) / msk.sum().clamp(min=1) # [H]
                sent_reps.append(rep)
            if not sent_reps:  # 防御
                sent_reps = [seq[i, 0]]
            reps = torch.stack(sent_reps)          # [S,H]
            logit = self.classifier(reps)          # [S,2]
            logits_list.append(logit)

        return torch.stack(logits_list, dim=0)     # [B,S,2]


class BertSentenceTagger:
    def __init__(self, model_dir: str, max_length: int = 512,
             device: Optional[str] = None, threshold: float = 0.5):
        """
        model_dir: 本地 HF 目录 或 .pth（state_dict）
        """
        assert model_dir is not None, "model_dir 不能为空（HF 目录或 .pth 路径）"

        # 1 加载模型和Tokenizer
        backbone_name = CONFIG.BERT_TOKENIZER
        num_labels = int(getattr(CONFIG, "NUM_LABELS", 2))

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        backbone = AutoModel.from_pretrained(
            backbone_name,
            config=AutoConfig.from_pretrained(backbone_name)
        )
        
        # 1.1 加载模型
        self.model = SentenceClassifier(backbone.config, backbone, self.tokenizer, num_labels=num_labels)

        # 1.2 加载模型和Tokenizer
        if os.path.isdir(model_dir):
            # 如果你把自定义头权重也保存成了同名键值（不一定能匹配），尝试读取 pytorch_model.bin
            pt_bin = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(pt_bin):
                state = torch.load(pt_bin, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
        elif model_dir.endswith(".pth"):
            state = torch.load(model_dir, map_location="cpu")
            # 兼容常见包裹
            for key in ("state_dict", "model_state_dict"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
            # 去前缀
            def _strip(k): 
                return k[len("module."):] if k.startswith("module.") else (k[len("model."):] if k.startswith("model.") else k)
            state = { _strip(k): v for k,v in state.items() }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:   core_logger.warning("[BERT] missing keys: %s", missing)
            if unexpected:core_logger.warning("[BERT] unexpected keys: %s", unexpected)
        else:
            raise ValueError(f"Unsupported model_dir: {model_dir}")

        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.max_length = int(max_length)
        self.threshold = float(threshold)
        self.sep_token = self.tokenizer.sep_token or "</s>"

    @torch.inference_mode()
    def predict_window(self, texts_in_window: List[str]) -> List[float]:
        """
        对一个滑动窗口（若干句子）进行推理，返回窗口内每个句子的 1 类概率。
        强制 batch_size=1（逐窗口）。
        """
        full = f" {self.sep_token} ".join(texts_in_window)
        enc = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc)[0]          # [S,2]
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        return probs

def label_sentences_via_bert(json_path: str, context_size: int = 2,
                             max_length: int = 512, threshold: Optional[float] = None,
                             min_sentences: int = 3) -> bool:
    """
    推理时也走“滑动窗口 + sep 拼接”，逐窗口(batch_size=1)推理；
    将每个窗口内各句子的概率聚合回全局句子（平均）。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容两种 schema
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and data["data"]:
            item = data["data"][0]
            sents = item.get("sentences", [])
            container = item
        else:
            sents = data.get("sentences", [])
            container = data

        if not sents or "text" not in sents[0]:
            raise RuntimeError(f"unexpected json schema in {json_path}")

        tagger = BertSentenceTagger(
            model_dir=model_dir,              # 你现有的变量/路径
            max_length=max_length,
            threshold=(threshold if threshold is not None else float(getattr(CONFIG, "BERT_THRESHOLD", 0.5)))
        )

        # 1) 构建与训练一致的窗口
        windows = _build_windows(
            sentences=sents,
            tokenizer=tagger.tokenizer,
            max_length=max_length,
            min_sentences=min_sentences,
            context_size=context_size
        )
        if not windows:
            # 至少保证一个窗口
            windows = [{"sentences": sents[:min(len(sents), min_sentences)], "indices": list(range(min(len(sents), min_sentences)))}]

        # 2) 逐窗口（batch_size=1）推理，并把概率聚合到全局句子上
        agg_sum = [0.0] * len(sents)
        agg_cnt = [0]   * len(sents)
        for win in windows:
            texts = [x["text"] for x in win["sentences"]]
            probs = tagger.predict_window(texts)      # len == len(texts)
            for local_i, global_i in enumerate(win["indices"]):
                p1 = float(probs[local_i])
                agg_sum[global_i] += p1
                agg_cnt[global_i] += 1

        thr = tagger.threshold
        for i in range(len(sents)):
            prob = (agg_sum[i] / agg_cnt[i]) if agg_cnt[i] else 0.0
            sents[i]["_bert_prob"] = round(prob, 6)
            sents[i]["label"] = int(prob >= thr)

        container["sentences"] = sents
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    except Exception as e:
        logger.exception("label_sentences_via_bert error: %s", e)
        return False


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
    if not core_logger.handlers:
        core_logger.addHandler(sh)
        core_logger.addHandler(fh)

def process_until_description(cat: str, entry: dict) -> dict:
    """
    处理单篇论文：下载 PDF -> GROBID -> TXT -> JSON -> LLM 贴标签 -> 汇总 data_description
    不做链接抽取与最终四列表。
    返回一个部分结果字典（也会写入 OUT_PROCESSED 下的临时 JSON）
    """
    arxiv_id = entry["id"]
    # day = day_str()
    day = ledger_day_name()

    # 目录与路径
    pdf_dir = os.path.join(CONFIG.OUT_PDF, day, cat)
    ensure_dir(pdf_dir)
    txt_path = os.path.join(pdf_dir, f"{arxiv_id}.txt")
    json_path = os.path.join(pdf_dir, f"{arxiv_id}.json")
    pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")

    #Ⅰ 下载 PDF（若已存在则跳过）
    if not os.path.exists(pdf_path):
        if not download_pdf(pdf_link_from_id(arxiv_id), pdf_path):
            raise RuntimeError(f"PDF download failed for {arxiv_id}")
        time.sleep(1.0 + random.random())

    #Ⅱ GROBID -> TXT
    if not os.path.exists(txt_path):
        if not grobid_extract_to_txt(pdf_path, txt_path):
            raise RuntimeError(f"GROBID failed for {arxiv_id}")

    #Ⅲ TXT -> JSON（句子切分，默认 label=0）
    if not os.path.exists(json_path):
        if not txt_to_json(txt_path, json_path):
            raise RuntimeError(f"txt_to_json failed for {arxiv_id}")

    #Ⅳ  BERT 贴标签（可调上下文窗口/阈值/最大长度）
    bert_ctx   = int(getattr(CONFIG, "SENT_CTX_SIZE", 2))        # 句子级上下文窗口（前后各 N 句）
    bert_max   = int(getattr(CONFIG, "SENT_MAX_LEN", 512))       # 句子级输入最大 token 长度
    bert_thr   = float(getattr(CONFIG, "BERT_SENT_THRESHOLD", getattr(CONFIG, "BERT_THRESHOLD", 0.5)))
    ok = label_sentences_via_bert(
        json_path=json_path,
        context_size=int(getattr(CONFIG, "SENT_CTX_SIZE", 2)),
        max_length=int(getattr(CONFIG, "SENT_MAX_LEN", 256)),
        threshold=float(getattr(CONFIG, "BERT_SENT_THRESHOLD", getattr(CONFIG, "BERT_THRESHOLD", 0.5)))
    )

    if not ok:
        raise RuntimeError(f"label_sentences_via_bert failed for {arxiv_id}")
    # 汇总数据集描述
    dataset_desc = collect_dataset_description(json_path)

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
        # === 新增：保存 BERT 打分与阈值、是否通过 ===
        "bert_gate": {
            "prob": float(entry.get("_bert_prob", -1.0)),
            "threshold": float(CONFIG.BERT_THRESHOLD),
            "accepted": bool(float(entry.get("_bert_prob", -1.0)) >= float(CONFIG.BERT_THRESHOLD))
        }
    }

    core_logger.info("[BERT] %s prob=%.3f thr=%.3f accepted=%s",
                arxiv_id,
                stage_record["bert_gate"]["prob"],
                stage_record["bert_gate"]["threshold"],
                stage_record["bert_gate"]["accepted"])

    stage_path = os.path.join(stage_dir, f"{arxiv_id}.partial.json")
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump(stage_record, f, ensure_ascii=False, indent=2)

    return stage_record


def _write_final_from_dataset_links(rec: dict, dataset_links):
    """
    将数据集链接写入的四列表 JSON（无链接则不写）。
    """
    if not dataset_links:
        return
    ts = now_iso()
    final_records = [{
        "Paper Link": rec["paper_link_pdf"],
        "Dataset Link": lk,
        "Dataset Description": rec.get("dataset_description", "") or "",
        "Timestamp": ts
    } for lk in dataset_links]
    append_final_records(final_records)


def _run_latex_stage(cat: str, results: list):
    """
    LaTeX 流程（加入“以 URL 为枢轴的滑动窗口 + 结构化 Prompt”）： 
      下载 -> 解压 -> 解析出所有链接(带上下文) -> 
      用滑动窗口构造 [CONTEXT] -> Qwen 分类 -> 
      写 dataset 链接 -> 旁路落地（含 window_sentences，便于后续 BERT 训练）
    """
    if not CONFIG.LATEX_ENABLED:
        core_logger.info("[LaTeX] disabled by CONFIG.LATEX_ENABLED")
        return

    # === 超参：可放进 CONFIG，提供兜底默认 ===
    bert_tok_name  = getattr(CONFIG, "BERT_TOKENIZER", "bert-base-uncased")
    max_length     = int(getattr(CONFIG, "BERT_MAX_LENGTH", 512))
    min_sentences  = int(getattr(CONFIG, "CTX_MIN_SENTENCES", 3))
    context_size   = int(getattr(CONFIG, "CTX_CONTEXT_SIZE", 2))
    # 解析 LaTeX 时的“探测窗口”，给我们提供足够的候选上下文用于收缩/扩张
    probe_window   = int(getattr(CONFIG, "CTX_PROBE_WINDOW", max(6, context_size * 3)))

    tokenizer = AutoTokenizer.from_pretrained(bert_tok_name)
    sep_token = tokenizer.sep_token or "</s>"

    day = ledger_day_name()
    have_src = 0
    total = len(results)

    for rec in results:
        aid = rec["arxiv_id"]
        tex_root = os.path.join(CONFIG.OUT_LATEX, day, cat, aid)
        tar_path = os.path.join(tex_root, f"{aid}.tar")
        src_dir = os.path.join(tex_root, "src")
        ensure_dir(tex_root)

        # 1) 下载源码
        ok_dl = download_latex_source(aid, tar_path)
        if not ok_dl:
            core_logger.info("[LaTeX] no source / download failed: %s", aid)
            continue
        have_src += 1

        # 2) 解压
        ok_ex = extract_tar(tar_path, src_dir)
        if not ok_ex:
            core_logger.warning("[LaTeX] extract failed: %s", aid)
            continue

        # 2.5) 生成展开后的纯文本副本
        flat_txt_path = os.path.join(tex_root, f"{aid}.latex.txt")
        _save_latex_plaintext(src_dir, flat_txt_path)
        core_logger.info("[LaTeX] saved plaintext: %s", flat_txt_path)

        # 3) 解析链接（加大探测窗口，为后续滑动窗口提供原材料）
        links_detail = parse_links_with_sentence_context(src_dir, window=probe_window)
        if not links_detail:
            core_logger.info("[LaTeX] no links parsed: %s", aid)
            continue

        # 4) 基于“URL 枢轴 + 滑动窗口”构造 Prompt
        qwen_inputs = []
        windows_meta = []  # 用于旁路落地
        for it in links_detail:
            before = it.get("ctx_before", []) or []
            after  = it.get("ctx_after", []) or []
            section = it.get("section", "")
            anchor  = it.get("anchor", "")
            url     = it["url"]

            # 用一个非常短的“枢轴占位句”把 URL 明确进上下文
            pivot_text = f"[URL_PIVOT] {anchor or url}"

            window_sents = _build_single_window_around_pivot(
                tokenizer=tokenizer,
                sep_token=sep_token,
                before_sents=before,
                after_sents=after,
                pivot_text=pivot_text,
                max_length=max_length,
                min_sentences=min_sentences,
                context_size=context_size
            )
            # 防御：万一空，至少保证一个 pivot
            if not window_sents:
                window_sents = [pivot_text]

            # 结构化 Prompt（模仿你给的风格）
            context_text = f" {sep_token} ".join(window_sents)
            qwen_inputs.append({
                "url": url,
                "input": (
                    f"[URL] {url}\n"
                    f"[ANCHOR] {anchor}\n"
                    f"[SECTION] {section}\n"
                    f"[CONTEXT]\n{context_text}"
                )
            })

            windows_meta.append({
                "url": url,
                "file": it.get("file", ""),
                "line": it.get("line", -1),
                "section": it.get("section", ""),
                "anchor": anchor,
                "window_sentences": window_sents,
                "sep_token": sep_token,
                "max_length": max_length,
                "min_sentences": min_sentences,
                "context_size": context_size
            })

        # 5) Qwen 初分类（基于 URL+窗口上下文）
        items = classify_link_inputs_via_llm(qwen_inputs)

        # 6) 取 dataset 链接 -> 写最终四列表
        ds_links = [it["url"] for it in items if it.get("label") == "dataset"]
        _write_final_from_dataset_links(rec, ds_links)

        # 7) 旁路产物（审计/训练样本）
        side = {
            "arxiv_id": aid,
            "paper_link_pdf": rec["paper_link_pdf"],
            "parsed_links": [x["url"] for x in links_detail],
            "classified_ctx": []
        }

        # 与原先 zip 对齐：仍然是 1:1
        for ld, qi, lab, meta in zip(links_detail, qwen_inputs, items, windows_meta):
            side["classified_ctx"].append({
                "url": ld["url"],
                "input": qi["input"],             # 结构化 Prompt
                "label": lab.get("label", "other"),
                "why": lab.get("why", ""),
                "file": ld.get("file", ""),
                "line": ld.get("line", -1),
                "section": ld.get("section", ""),
                "anchor": ld.get("anchor", ""),
                # 保留原探测上下文，方便回溯
                "ctx_before": ld.get("ctx_before", []),
                "ctx_after": ld.get("ctx_after", []),
                # 新增：滑动窗口后的最终句子序列，便于直接做 BERT 训练
                "window_sentences": meta["window_sentences"],
                "sep_token": meta["sep_token"],
                "max_length": meta["max_length"],
                "min_sentences": meta["min_sentences"],
                "context_size": meta["context_size"],
            })

        side_path = os.path.join(tex_root, f"{aid}.links.json")
        with open(side_path, "w", encoding="utf-8") as f:
            json.dump(side, f, ensure_ascii=False, indent=2)

        core_logger.info(
            "[LaTeX] %s: parsed=%d, dataset=%d (tok_win: max=%d, min_sent=%d, ctx=%d)",
            aid, len(links_detail), len(ds_links), max_length, min_sentences, context_size
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


def append_link_corpus(records: List[Dict]):
    """
    记录每条链接的分类，用于构建“分类数据集”。
    记录字段示例：
    {
      "Arxiv ID": "2509.01234v1",
      "Category": "cs.CL",
      "Paper Link": "https://arxiv.org/pdf/2509.01234v1.pdf",
      "URL": "https://github.com/xxx/yyy",
      "Label": "dataset|code|other",
      "Why": "Qwen 的分类理由（可空）",
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
        # 幂等去重（Paper Link + URL + Label）
        merged = existing + (records or [])
        seen, dedup = set(), []
        for r in merged:
            key = (r.get("Paper Link",""), r.get("URL",""), r.get("Label",""))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dedup, f, ensure_ascii=False, indent=2)
    core_logger.info("Link corpus written: %s (+%d)", out_path, len(records))

