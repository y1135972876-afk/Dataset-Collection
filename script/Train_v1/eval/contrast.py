#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在同一测试集上对比：
(1) 已训练好的 cased 微调模型
(2) 原始 bert-base-uncased（未训练，随机分类头）

会分别输出两套评测结果与分类报告，并保存对比表。
"""

import os
import sys
import json
import argparse
import contextlib
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from sklearn.metrics import classification_report

# ====== 按你的项目路径修改 ======
sys.path.append("/home/kzlab/muse/Savvy/Data_collection/dataset/dataset-descrption/code")

from correct_bert import (
    PaperDataset,
    BertForSentenceClassification,
    custom_collate_fn,
    calculate_metrics,
    load_data,
    save_predictions_to_json
)

# -----------------------------
# 评测函数（含若干健壮性与性能优化）
# -----------------------------
def evaluate_model(model, test_dataloader, device) -> Dict[str, Any]:
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0.0
    predictions_data = []

    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            paper_names = batch['paper_names']
            sentences = batch['sentences']
            indices = batch['indices']

            autocast_ctx = torch.cuda.amp.autocast if device.type == 'cuda' else contextlib.nullcontext
            with autocast_ctx():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            loss, logits = outputs
            test_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)           # [B, max_sent]
            probs = torch.softmax(logits, dim=-1)                # [B, max_sent, C]

            for b in range(len(paper_names)):
                valid_indices = indices[b][indices[b] != -1]
                valid_predictions = predictions[b][:len(valid_indices)]
                valid_probs = probs[b][:len(valid_indices)]
                valid_labels = labels[b][:len(valid_indices)]
                valid_sentences = sentences[b][:len(valid_indices)]

                sentences_data = []
                for idx, pred, prob, label, sent_text in zip(
                    valid_indices, valid_predictions, valid_probs, valid_labels, valid_sentences
                ):
                    if label != -100:
                        sentences_data.append({
                            'sentence_index': idx.item(),
                            'sentence_text': sent_text,
                            'predicted_label': pred.item(),
                            'true_label': label.item(),
                            'confidence': prob[pred].item(),
                            'probabilities': {
                                'class_0': prob[0].item(),
                                'class_1': prob[1].item()
                            }
                        })
                        all_predictions.append(pred.item())
                        all_labels.append(label.item())

                if sentences_data:
                    predictions_data.append({
                        'paper_name': paper_names[b],
                        'sentences': sentences_data
                    })

    avg_loss = test_loss / max(1, len(test_dataloader))

    # 空集保护
    if len(all_labels) == 0:
        return {
            'loss': avg_loss,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'report': "No valid samples to evaluate.",
            'predictions': predictions_data
        }

    # 注意：calculate_metrics 通常期望 (y_true, y_pred)
    accuracy, precision, recall, f1 = calculate_metrics(
        torch.tensor(all_labels),
        torch.tensor(all_predictions)
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=['Class 0', 'Class 1'],
        digits=4,
        zero_division=0
    )

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'predictions': predictions_data
    }

# -----------------------------
# 构建 DataLoader（给定 tokenizer）
# -----------------------------
def build_dataloader(papers, tokenizer, device, batch_size=1):
    dataset = PaperDataset(
        papers=papers,
        tokenizer=tokenizer,
        max_length=512,
        min_sentences=3,
        context_size=2
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=max(0, (os.cpu_count() or 1) // 2),
        pin_memory=(device.type == 'cuda')
    )
    return loader

# -----------------------------
# 主流程：对比评测
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="对比：微调模型 vs 原始 bert-base-uncased")
    parser.add_argument("--test_file", type=str,
                        default="/home/kzlab/muse/Savvy/Data_collection/dataset/test_dataset.json",
                        help="测试集 JSON 路径")
    parser.add_argument("--finetuned_model_path", type=str,
                        default="/home/kzlab/muse/Savvy/Data_collection/output/fuxian/1224/best_model.pth",
                        help="已训练模型权重路径（.pth）")
    parser.add_argument("--bert_base_cased_path", type=str,
                        default="/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased",
                        help="用于微调模型的 tokenizer/config 基座路径")
    parser.add_argument("--bert_base_uncased_path", type=str,
                        default="/home/kzlab/muse/Savvy/Data_collection/models/bert-base-uncased",
                        help="原始 uncased 模型的路径（可用本地或 HF 名称 'bert-base-uncased'）")
    parser.add_argument("--output_dir", type=str,
                        default="/home/kzlab/muse/Savvy/Data_collection/output/compare_finetuned_vs_uncased",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="评测 batch_size（建议=1 保持与原实现一致）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # 1) 加载测试数据（一次）
    print("[Info] Loading test data...")
    papers = load_data(args.test_file)
    print(f"[Info] Loaded {len(papers)} test papers.")

    # ========== A. 微调模型 (cased) ==========
    print("\n[Part A] Evaluating FINETUNED (cased) model...")
    tok_cased = BertTokenizer.from_pretrained(args.bert_base_cased_path)
    cfg_cased = BertConfig.from_pretrained(args.bert_base_cased_path, num_labels=2)
    finetuned_model = BertForSentenceClassification(cfg_cased, tokenizer=tok_cased)
    finetuned_model.load_state_dict(torch.load(args.finetuned_model_path, map_location=device))
    finetuned_model.to(device)

    dl_cased = build_dataloader(papers, tok_cased, device, batch_size=args.batch_size)
    res_finetuned = evaluate_model(finetuned_model, dl_cased, device)

    # 保存微调模型结果
    finetuned_dir = os.path.join(args.output_dir, "finetuned_cased")
    os.makedirs(finetuned_dir, exist_ok=True)
    with open(os.path.join(finetuned_dir, "metrics.json"), "w") as f:
        json.dump({
            'loss': res_finetuned['loss'],
            'accuracy': res_finetuned['accuracy'],
            'precision': res_finetuned['precision'],
            'recall': res_finetuned['recall'],
            'f1': res_finetuned['f1'],
            'classification_report': res_finetuned['report']
        }, f, indent=2, ensure_ascii=False)
    save_predictions_to_json(res_finetuned['predictions'], os.path.join(finetuned_dir, "predictions.json"))

    # ========== B. 原始 uncased 模型（未训练） ==========
    print("\n[Part B] Evaluating BASELINE (bert-base-uncased, no fine-tune)...")
    # 支持本地或直接从 HF 名称加载
    tok_uncased = BertTokenizer.from_pretrained(args.bert_base_uncased_path)
    cfg_uncased = BertConfig.from_pretrained(args.bert_base_uncased_path, num_labels=2)
    baseline_model = BertForSentenceClassification(cfg_uncased, tokenizer=tok_uncased)  # 分类头随机初始化
    baseline_model.to(device)

    dl_uncased = build_dataloader(papers, tok_uncased, device, batch_size=args.batch_size)
    res_baseline = evaluate_model(baseline_model, dl_uncased, device)

    # 保存 baseline 结果
    baseline_dir = os.path.join(args.output_dir, "baseline_uncased")
    os.makedirs(baseline_dir, exist_ok=True)
    with open(os.path.join(baseline_dir, "metrics.json"), "w") as f:
        json.dump({
            'loss': res_baseline['loss'],
            'accuracy': res_baseline['accuracy'],
            'precision': res_baseline['precision'],
            'recall': res_baseline['recall'],
            'f1': res_baseline['f1'],
            'classification_report': res_baseline['report']
        }, f, indent=2, ensure_ascii=False)
    save_predictions_to_json(res_baseline['predictions'], os.path.join(baseline_dir, "predictions.json"))

    # ========== 汇总对比 ==========
    compare = {
        "finetuned_cased": {
            "loss": res_finetuned['loss'],
            "accuracy": res_finetuned['accuracy'],
            "precision": res_finetuned['precision'],
            "recall": res_finetuned['recall'],
            "f1": res_finetuned['f1'],
        },
        "baseline_uncased": {
            "loss": res_baseline['loss'],
            "accuracy": res_baseline['accuracy'],
            "precision": res_baseline['precision'],
            "recall": res_baseline['recall'],
            "f1": res_baseline['f1'],
        }
    }
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)

    # 友好打印
    def fmt(m): return f"Loss={m['loss']:.4f} | Acc={m['accuracy']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f}"
    print("\n========== Comparison ==========")
    print("FINETUNED (cased):   ", fmt(compare["finetuned_cased"]))
    print("BASELINE  (uncased): ", fmt(compare["baseline_uncased"]))
    print(f"\n[Saved] Results to: {args.output_dir}")
    print(f"- {finetuned_dir}/metrics.json, predictions.json")
    print(f"- {baseline_dir}/metrics.json, predictions.json")
    print(f"- {args.output_dir}/comparison.json")

if __name__ == "__main__":
    main()
