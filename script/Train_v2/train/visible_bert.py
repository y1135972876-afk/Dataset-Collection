# train.py
import os
import json
import random
from datetime import datetime
import html as html_lib

import numpy as np
import torch
from torch import nn

from utils.utils_train import (
    set_seed, get_device, load_json_papers,
    build_tokenizer, build_backbone_model, SentenceClassifier,
    make_dataloader, evaluate,
    save_predictions_to_json, make_experiment_dir
)


# -----------------------------
# 工具：尝试加载分类头（classifier_head.bin）
# -----------------------------
def _maybe_load_classifier_head(model: SentenceClassifier, head_dir: str):
    if not head_dir:
        return
    head_path = os.path.join(head_dir, "classifier_head.bin")
    if os.path.exists(head_path):
        head_state = torch.load(head_path, map_location="cpu")
        missing, unexpected = model.classifier.load_state_dict(head_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"classifier_head.bin mismatch: missing={missing}, unexpected={unexpected}")
        print(f"[OK] classifier head loaded from: {head_path}")
    else:
        print(f"[WARN] classifier_head.bin not found in: {head_dir} (using randomly initialized head)")


# -----------------------------
# 工具：从 evaluate 的结构化结果拍平为句子级列表
# -----------------------------
def _flatten_predictions_for_sampling(preds_struct):
    flat = []
    for paper in preds_struct:
        paper_name = paper.get("paper_name", "")
        for s in paper.get("sentences", []):
            prob_0 = float(s.get("probabilities", {}).get("class_0", 0.0))
            prob_1 = float(s.get("probabilities", {}).get("class_1", 0.0))
            pred   = int(s.get("predicted_label", 0))
            true   = s.get("true_label", None)
            conf   = float(s.get("confidence", max(prob_0, prob_1)))
            text   = s.get("sentence_text", "")
            idx    = int(s.get("sentence_index", -1))
            flat.append({
                "paper_name": paper_name,
                "sentence_index": idx,
                "predicted_label": pred,
                "true_label": true,
                "prob_0": prob_0,
                "prob_1": prob_1,
                "confidence": conf,
                "sentence_text": text
            })
    return flat


# -----------------------------
# 颜色映射：单色红（值越大越深）
# -----------------------------
def _color_value_abs(v: float) -> str:
    v = max(0.0, min(1.0, float(v)))
    sat   = 30 + 55 * v   # 30% -> 85%
    light = 97 - 42 * v   # 97% -> 55%
    return f"hsl(0,{sat:.0f}%,{light:.0f}%)"


# -----------------------------
# 词级显著性：grad*input / grad-norm（二选一）
# 这里默认 grad*input（signed=False 即取绝对值）
# -----------------------------
def _token_attr_for_single_sentence(
    model: SentenceClassifier,
    tokenizer,
    device,
    text: str,
    max_length: int = 512,
    class_index: int = 1,           # 正类
    method: str = "grad_input",     # "grad_input" or "grad_norm"
    signed: bool = False            # 默认取绝对值做高亮
):
    """
    对“单句”做一次显著性分析：
      - 构造 [CLS] + tokens + [SEP] 输入
      - 用 backbone 得到 last_hidden_state
      - 按训练时逻辑对 CLS..SEP 片段做平均池化，过分类头得到 logits
      - 对 softmax(class_index) 反传，取 token 级的 grad*input（或 grad-norm）
      - 返回 token 列表与 0~1 归一化的分数，以及概率/预测
    """
    # 构造 ids/attention
    ids = [tokenizer.cls_token_id]
    piece = tokenizer.encode(text, add_special_tokens=False)
    ids.extend(piece)
    ids.append(tokenizer.sep_token_id)
    if len(ids) > max_length:
        ids = ids[:max_length]
        if ids[-1] != tokenizer.sep_token_id:
            ids[-1] = tokenizer.sep_token_id
    attn = [1] * len(ids)
    if len(ids) < max_length:
        pad = [tokenizer.pad_token_id] * (max_length - len(ids))
        pad_m = [0] * (max_length - len(ids))
        ids = ids + pad
        attn = attn + pad_m

    input_ids      = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attn], dtype=torch.long, device=device)

    # 前向（直接用 backbone，按照 SentenceClassifier 的 pooling 方式手工做一遍）
    outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    seq = outputs.last_hidden_state            # [1, L, H]
    seq.retain_grad()

    # 边界：CLS + 首个 SEP
    sep_id = tokenizer.sep_token_id
    sep_pos = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0]
    if len(sep_pos) == 0:
        # 极端情况无 SEP：回退到 [CLS] 单点
        st, ed = 0, 0
    else:
        st, ed = 0, int(sep_pos[0].item())

    tok = seq[:, st:ed+1, :]                                 # [1, T, H]
    msk = attention_mask[:, st:ed+1].unsqueeze(-1).float()   # [1, T, 1]

    rep = (tok * msk).sum(dim=1) / msk.sum(dim=1).clamp_min(1.0)  # [1, H]
    logits = model.classifier(rep)                                # [1, 2]
    probs = torch.softmax(logits, dim=-1)[0]                      # [2]
    pred  = int(torch.argmax(probs).item())
    p1    = float(probs[class_index].item())

    # 反传
    model.classifier.zero_grad(set_to_none=True)
    if seq.grad is not None:
        seq.grad.zero_()
    probs[class_index].backward(retain_graph=False)

    g = seq.grad[:, st:ed+1, :]   # [1, T, H]
    x = seq[:, st:ed+1, :]        # [1, T, H]

    if method == "grad_input":
        raw = (g * x).sum(dim=-1).squeeze(0)   # [T]，有符号
        scores = raw if signed else raw.abs()
    else:
        scores = g.norm(dim=-1).squeeze(0)     # [T]，非负

    # 归一化到 [0,1]
    scores = scores.detach().cpu().numpy().astype(np.float32)
    if scores.size > 0:
        mn, mx = float(scores.min()), float(scores.max())
        if mx - mn > 1e-12:
            scores = (scores - mn) / (mx - mn)
        else:
            scores = np.zeros_like(scores)
    tokens = tokenizer.convert_ids_to_tokens(ids[st:ed+1])

    return {
        "tokens": tokens,            # 含 [CLS] 和 [SEP]
        "scores": scores.tolist(),   # 0~1
        "prob_1": p1,
        "pred": pred
    }


# -----------------------------
# HTML 渲染（句级 + 词级）
# -----------------------------
def _render_html_with_tokens(sent_list, out_path, threshold=0.5):
    """
    sent_list 每项包含：
      paper_name, sentence_index, predicted_label, true_label, prob_1, sentence_text,
      token_vis: {tokens: [...], scores: [...], pred, prob_1}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 统计 sample 内部准确率（若有 gold）
    n_with_gold = sum(1 for s in sent_list if s.get("true_label") in (0, 1))
    n_correct   = sum(1 for s in sent_list
                      if s.get("true_label") in (0, 1) and s["predicted_label"] == int(s["true_label"]))
    acc = (n_correct / n_with_gold) if n_with_gold > 0 else 0.0

    style = """
    <style>
      body{font-family:ui-sans-serif,Arial; line-height:1.5; padding:20px;}
      .meta{color:#555; margin-bottom:12px;}
      table{border-collapse:collapse; width:100%;}
      th, td{border:1px solid #ddd; padding:8px; vertical-align:top;}
      th{background:#f7f7f7;}
      .ok{background:#eaffea;}
      .bad{background:#ffecec;}
      .chip{display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#eee; margin-left:6px;}
      .probbar{display:inline-block; height:10px; background:#eaeaea; width:160px; vertical-align:middle; margin:0 8px;}
      .probbar > span{display:block; height:10px; background:#1a73e8;}
      .sent{white-space:pre-wrap;}
      .tokline{margin-top:6px;}
      .tok{display:inline-block; margin:2px 3px; padding:2px 4px; border-radius:3px;}
      .tok.gray{background:#f0f0f0;}
      .minor{color:#666; font-size:12px;}
    </style>
    """

    rows = []
    rows.append("<tr><th>#</th><th>Paper</th><th>Idx</th><th>Pred</th><th>Prob(1)</th><th>Gold</th><th>Correct?</th><th>Sentence & Tokens</th></tr>")

    for i, s in enumerate(sent_list, 1):
        pred = int(s["predicted_label"])
        p1   = float(s["prob_1"])
        gold = s.get("true_label")
        correct = (gold in (0,1) and pred == int(gold))
        cls = "ok" if correct else ("bad" if gold in (0,1) else "")
        bar_w = max(0.0, min(1.0, p1)) * 160.0

        # token 可视化
        tok_html_bits = []
        tv = s.get("token_vis")
        if tv and tv.get("tokens") and tv.get("scores"):
            tt = tv["tokens"]
            ss = tv["scores"]
            for t, sc in zip(tt, ss):
                if t in ("[SEP]", "</s>"):
                    tok_html_bits.append(f"<span class='tok gray'>{html_lib.escape(t)}</span>")
                else:
                    color = _color_value_abs(sc)
                    tok_html_bits.append(
                        f"<span class='tok' style='background:{color};'>{html_lib.escape(t)}</span>"
                    )
        tok_html = "<div class='tokline'>" + "".join(tok_html_bits) + "</div>"

        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{i}</td>"
            f"<td>{html_lib.escape(s.get('paper_name',''))}</td>"
            f"<td>{s.get('sentence_index',-1)}</td>"
            f"<td>{pred}</td>"
            f"<td>{p1:.3f}<span class='probbar'><span style='width:{bar_w:.0f}px'></span></span></td>"
            f"<td>{'' if gold is None else gold}</td>"
            f"<td>{'' if gold is None else ('✓' if correct else '✗')}</td>"
            f"<td>"
              f"<div class='sent'>{html_lib.escape(s.get('sentence_text',''))}</div>"
              f"{tok_html}"
              f"<div class='minor'>Token coloring = grad×input 强度（绝对值），句内归一化到 [0,1]；"
              f"灰色表示 [SEP] 等无意义标记。</div>"
            f"</td>"
            f"</tr>"
        )

    html = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        "<title>BERT Sentence Inspection (with Token Importance)</title>",
        style,
        "</head><body>",
        "<h2>BERT Sentence Inspection (Sentence + Token-level)</h2>",
        f"<div class='meta'>Sample size: <b>{len(sent_list)}</b> | Threshold={threshold:.2f} "
        f"| With Gold: {n_with_gold} | Accuracy on gold: {acc:.3f}</div>",
        "<table>",
        *rows,
        "</table>",
        "</body></html>"
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[OK] HTML saved -> {out_path}")


def main():
    # ---------------------------- 配置区 ----------------------------
    MODEL_PATH = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs/bert-base-cased__train_val_revise.json__20251025_230335__exp1/best_hf"
    RESUME_HF_DIR = None  # 若想显式指定另一个 hf 目录，可填这里

    VAL_JSON   = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json"
    SEED       = 42
    OUT_ROOT   = "/home/kzlab/muse/Savvy/Data_collection/script/Train_v2/train/outputs"
    TAG        = "exp1"

    BATCH_SIZE = 1
    MAX_LENGTH = 512
    MIN_SENT   = 3
    CONTEXT    = 2

    # HTML 输出目录（你要求的）
    HTML_DIR   = "/home/kzlab/muse/Savvy/Data_collection/temp/inspectionBert"
    SAMPLE_N   = 100   # 抽样 100 句

    set_seed(SEED)
    device = get_device()

    # 1) 数据加载
    print("Loading data ...")
    val_papers = load_json_papers(VAL_JSON)
    print(f"Validation set size: {len(val_papers)}")

    # 2) 模型与 tokenizer 初始化
    if RESUME_HF_DIR:
        tokenizer = build_tokenizer(RESUME_HF_DIR)
        backbone  = build_backbone_model(RESUME_HF_DIR)
        head_dir_for_load = RESUME_HF_DIR
    else:
        tokenizer = build_tokenizer(MODEL_PATH)
        backbone  = build_backbone_model(MODEL_PATH)
        head_dir_for_load = MODEL_PATH  # 也尝试从这里加载分类头

    model = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2)
    _maybe_load_classifier_head(model, head_dir_for_load)
    model = model.to(device)
    model.eval()   # 可视化时不需要 dropout

    # 3) DataLoader（评估）
    val_loader = make_dataloader(
        val_papers, tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        min_sentences=MIN_SENT,
        context_size=CONTEXT,
        shuffle=False
    )

    # 4) 输出目录
    save_dir = make_experiment_dir(
        OUT_ROOT,
        RESUME_HF_DIR if RESUME_HF_DIR else MODEL_PATH,
        dataset_tag=os.path.basename(VAL_JSON),
        tag=TAG
    )

    # 5) 评估并保存结构化预测（保持原逻辑）
    va, preds = evaluate(model, val_loader, device)
    print(f"Eval metrics: {va}")
    save_predictions_to_json(preds, os.path.join(save_dir, "predictions", "preds_eval.json"))

    # 6) 拍平句子级结果，抽样 100 句
    flat = _flatten_predictions_for_sampling(preds)
    if not flat:
        print("[WARN] No sentence predictions to render.")
        return

    rng = random.Random(SEED)
    if len(flat) > SAMPLE_N:
        sample = rng.sample(flat, SAMPLE_N)
    else:
        sample = flat

    # 7) 对抽样的每句做一次 token 级显著性（grad×input），与句级结果一起渲染 HTML
    for s in sample:
        text = s.get("sentence_text", "")
        if not text:
            s["token_vis"] = {"tokens": [], "scores": [], "pred": s["predicted_label"], "prob_1": s["prob_1"]}
            continue
        tv = _token_attr_for_single_sentence(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=text,
            max_length=MAX_LENGTH,
            class_index=1,
            method="grad_input",
            signed=False  # 单色红系用绝对值强度
        )
        s["token_vis"] = tv

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(HTML_DIR, f"inspect_tokens_{ts}.html")
    _render_html_with_tokens(sample, out_html, threshold=0.5)


if __name__ == "__main__":
    main()
