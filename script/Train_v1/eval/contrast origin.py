# -*- coding: utf-8 -*-
"""
Self-contained evaluation script:
- Compare pretrained (un-finetuned) vs finetuned (best_model.pth) on test set
- Save metrics (acc/prec/recall/F1), confusion matrix, and per-sentence predictions
"""

# ==== Imports (must be at top) ====
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional   # ← 兼容 Py3.8/3.9 的 Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# ==== Config (you can edit) ====
DEFAULT_BERT_PATH = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"
DEFAULT_TEST_FILE = "/home/kzlab/muse/Savvy/Data_collection/dataset/test_dataset.json"
DEFAULT_OUTPUT_ROOT = "/home/kzlab/muse/Savvy/Data_collection/output/fuxian"

# Optional date string just for default paths (not required)
date_str = "1224"


# =========================
# Utilities & Dataset
# =========================
def calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """计算 acc/prec/recall/f1；忽略 label=-100；兼容GPU张量"""
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    if predictions.is_cuda:
        predictions_cpu = predictions.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
    else:
        predictions_cpu = predictions.numpy()
        labels_cpu = labels.numpy()

    if len(predictions_cpu) == 0 or len(labels_cpu) == 0:
        return 0.0, 0.0, 0.0, 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_cpu, predictions_cpu, average="binary", zero_division=0
    )
    accuracy = (predictions == labels).float().mean().item()
    return accuracy, precision, recall, f1


class PaperDataset(Dataset):
    """将论文-句子结构展开为窗口级样本；与训练版保持一致"""

    def __init__(self, papers, tokenizer, max_length=512, min_sentences=3, context_size=2):
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_sentences = min_sentences
        self.context_size = context_size
        self.flat_examples = self._prepare_examples()

    def _get_window_size(self, sentences, start_idx):
        current_tokens = 0
        window_size = 0
        for i in range(start_idx, len(sentences)):
            tokens = self.tokenizer.encode(sentences[i]['text'], add_special_tokens=False)
            next_length = current_tokens + len(tokens) + (1 if current_tokens > 0 else 2)
            if next_length >= self.max_length - 2:
                break
            current_tokens = next_length
            window_size += 1
        return max(self.min_sentences, window_size)

    def _prepare_examples(self):
        examples = []
        for paper in self.papers:
            sentences = paper['sentences']
            i = 0
            while i < len(sentences):
                window_size = self._get_window_size(sentences, i)
                end_idx = min(i + window_size, len(sentences))
                current_window = sentences[i:end_idx]
                current_indices = list(range(i, end_idx))

                window_tokens = sum(len(self.tokenizer.encode(sent['text'], add_special_tokens=False))
                                    for sent in current_window)
                has_positive = any(sent['label'] == 1 for sent in current_window)

                if has_positive:
                    first_positive = next(idx for idx, sent in enumerate(current_window) if sent['label'] == 1)
                    start_offset = max(0, first_positive - self.context_size)
                    end_offset = min(len(current_window), first_positive + self.context_size + 1)
                    current_window = current_window[start_offset:end_offset]
                    current_indices = current_indices[start_offset:end_offset]
                    window_tokens = sum(len(self.tokenizer.encode(sent['text'], add_special_tokens=False))
                                        for sent in current_window)

                if len(current_window) >= self.min_sentences:
                    examples.append({
                        'sentences': current_window,
                        'indices': current_indices,
                        'paper_name': paper['paper_name'],
                        'window_tokens': window_tokens,
                        'has_positive': has_positive
                    })

                step_size = max(1, window_size // 2) if has_positive else window_size // 2
                i += max(1, step_size)
        return examples

    def __len__(self):
        return len(self.flat_examples)

    def __getitem__(self, idx):
        example = self.flat_examples[idx]
        sentences = example['sentences']
        text_list = [s['text'] for s in sentences]
        labels = [s['label'] for s in sentences]

        full_text = ' [SEP] '.join(text_list)  # 注意：这里是普通字符串的 [SEP]

        enc = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.tensor(example['indices'], dtype=torch.long),
            'paper_name': example['paper_name'],
            'has_positive': example['has_positive'],
            'sentences': text_list
        }


def custom_collate_fn(batch):
    input_ids, attention_masks, labels, indices = [], [], [], []
    paper_names, sentences_list = [], []
    max_sentences = max(len(item['labels']) for item in batch)

    for item in batch:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])

        curr_labels = item['labels']
        if len(curr_labels) < max_sentences:
            pad = torch.full((max_sentences - len(curr_labels),), -100, dtype=curr_labels.dtype)
            curr_labels = torch.cat([curr_labels, pad])
        labels.append(curr_labels)

        curr_indices = item['indices']
        if len(curr_indices) < max_sentences:
            pad = torch.full((max_sentences - len(curr_indices),), -1, dtype=curr_indices.dtype)
            curr_indices = torch.cat([curr_indices, pad])
        indices.append(curr_indices)

        paper_names.append(item['paper_name'])
        sentences_list.append(item['sentences'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
        'indices': torch.stack(indices),
        'paper_names': paper_names,
        'sentences': sentences_list
    }


class BertForSentenceClassification(BertForSequenceClassification):
    """在 BertForSequenceClassification 基础上，改为句级分类（与训练版保持一致）"""

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer

        self.sentence_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
        )
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        device = next(self.parameters()).device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [B, L, H]

        batch_size = input_ids.size(0)
        if labels is not None:
            max_sentences = labels.size(1)
        else:
            max_sentences = max(
                len((input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]) + 1
                for i in range(batch_size)
            )

        logits_list = []
        for i in range(batch_size):
            sep_positions = (input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            cls_position = torch.tensor([0], device=input_ids.device)
            all_positions = torch.cat([cls_position, sep_positions])

            sentence_reps = []
            for j in range(len(all_positions) - 1):
                start_pos = all_positions[j]
                end_pos = all_positions[j + 1]
                toks = sequence_output[i, start_pos:end_pos + 1]
                sent_mask = attention_mask[i, start_pos:end_pos + 1].unsqueeze(-1)
                masked = toks * sent_mask
                rep = masked.sum(dim=0) / (sent_mask.sum() + 1e-10)
                sentence_reps.append(rep)

            if sentence_reps:
                sent_tensor = torch.stack(sentence_reps)
                current_n = len(sentence_reps)
                if current_n < max_sentences:
                    pad = torch.zeros(max_sentences - current_n, sent_tensor.size(-1), device=device)
                    sent_tensor = torch.cat([sent_tensor, pad], dim=0)
                else:
                    sent_tensor = sent_tensor[:max_sentences]

                sent_logits = self.classifier(sent_tensor)  # [num_sentences, 2]
                logits_list.append(sent_logits)

        padded_logits = torch.stack(logits_list)  # [B, S, 2]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(padded_logits.view(-1, 2), labels.view(-1))
        return loss, padded_logits


def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw['data']
    papers, current = [], None
    for item in sorted(data, key=lambda x: (x['paper_name'], x['section'], x['paragraph_id'])):
        if current is None or item['paper_name'] != current['paper_name']:
            if current is not None:
                papers.append(current)
            current = {'paper_name': item['paper_name'], 'sentences': []}
        current['sentences'].extend(item['sentences'])
    if current is not None:
        papers.append(current)
    return papers


def collect_predictions(model, dataloader, device):
    model.eval()
    predictions_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Collecting predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            indices = batch['indices']
            paper_names = batch['paper_names']
            sentences = batch['sentences']

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            for b in range(len(paper_names)):
                valid_indices = indices[b][indices[b] != -1]
                valid_preds = preds[b][:len(valid_indices)]
                valid_probs = probs[b][:len(valid_indices)]
                valid_labels = labels[b][:len(valid_indices)]
                valid_sentences = sentences[b][:len(valid_indices)]

                sent_data = []
                for idx, pred, prob, lab, sent_text in zip(
                    valid_indices, valid_preds, valid_probs, valid_labels, valid_sentences
                ):
                    if lab != -100:
                        sent_data.append({
                            'sentence_index': idx.item(),
                            'sentence_text': sent_text,
                            'predicted_label': pred.item(),
                            'true_label': lab.item(),
                            'confidence': prob[pred].item(),
                            'probabilities': {
                                'class_0': prob[0].item(),
                                'class_1': prob[1].item()
                            }
                        })
                if sent_data:
                    predictions_data.append({'paper_name': paper_names[b], 'sentences': sent_data})
    return predictions_data


def save_predictions_to_json(predictions_data, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    total_sentences = sum(len(p['sentences']) for p in predictions_data)
    correct = sum(1 for p in predictions_data for s in p['sentences'] if s['predicted_label'] == s['true_label'])
    total_positive = sum(1 for p in predictions_data for s in p['sentences'] if s['true_label'] == 1)

    output_data = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'num_papers': len(predictions_data),
            'total_sentences': total_sentences,
            'statistics': {
                'accuracy': correct / total_sentences if total_sentences > 0 else 0.0,
                'total_correct': correct,
                'total_positive_samples': total_positive,
                'positive_ratio': total_positive / total_sentences if total_sentences > 0 else 0.0
            }
        },
        'predictions': predictions_data
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Predictions -> {output_path}")


# =========================
# Evaluation helpers
# =========================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval on test'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if loss is not None:
                losses.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.view(-1))
            all_labels.append(labels.view(-1))

    if len(all_preds) == 0:
        return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]], 'report': {}}

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels)

    mask = all_labels != -100
    y_true = all_labels[mask].detach().cpu().numpy()
    y_pred = all_preds[mask].detach().cpu().numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    rep = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=['class_0', 'class_1'],
        zero_division=0, output_dict=True
    )
    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1': float(f1),
        'confusion_matrix': cm, 'report': rep
    }


def make_test_loader(tokenizer, test_file_path, batch_size=1, max_length=512):
    test_papers = load_data(test_file_path)
    test_dataset = PaperDataset(
        papers=test_papers, tokenizer=tokenizer,
        max_length=max_length, min_sentences=3, context_size=2
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return test_loader, len(test_papers)


def find_latest_ckpt(root=DEFAULT_OUTPUT_ROOT):
    root = Path(root)
    if not root.exists():
        return None
    cands = []
    for sub in root.iterdir():
        p = sub / "best_model.pth"
        if p.exists():
            cands.append(p)
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None


def evaluate_pre_and_post(
    bert_path: str = DEFAULT_BERT_PATH,
    test_file: str = DEFAULT_TEST_FILE,
    ckpt_path: str = "",
    batch_size: int = 1,
    out_dir_base: Optional[str] = None   # ← 用 Optional[str] 替代 str | None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")

    # Determine output dir
    if not ckpt_path:
        ckpt_path = find_latest_ckpt()
    if out_dir_base is None:
        out_dir_base = str(Path(ckpt_path).parent) if ckpt_path else os.path.join(DEFAULT_OUTPUT_ROOT, "eval_default")

    out_dir = os.path.join(out_dir_base, "test_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer & Config
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    config = BertConfig.from_pretrained(bert_path, num_labels=2)

    # Data
    test_loader, num_papers = make_test_loader(tokenizer, test_file, batch_size=batch_size)
    print(f"[Eval] Loaded test set: {num_papers} papers")

    # 1) Baseline (pretrained)
    baseline_model = BertForSentenceClassification.from_pretrained(
        bert_path, config=config, tokenizer=tokenizer
    ).to(device)
    print("[Eval] Evaluating baseline (pretrained, un-finetuned) ...")
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    baseline_preds = collect_predictions(baseline_model, test_loader, device)
    save_predictions_to_json(baseline_preds, os.path.join(out_dir, "test_predictions_pretrained.json"))

    # 2) Finetuned
    finetuned_model = BertForSentenceClassification.from_pretrained(
        bert_path, config=config, tokenizer=tokenizer
    ).to(device)

    if not ckpt_path or not os.path.exists(ckpt_path):
        # Fall back to date_str if present
        candidate = os.path.join(DEFAULT_OUTPUT_ROOT, date_str, "best_model.pth")
        if os.path.exists(candidate):
            ckpt_path = candidate
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 finetuned 权重：{ckpt_path}\n"
            f"请使用 --ckpt_path 明确传入，或确保 {DEFAULT_OUTPUT_ROOT}/*/best_model.pth 存在。"
        )

    print(f"[Eval] Loading finetuned weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    finetuned_model.load_state_dict(state_dict)

    print("[Eval] Evaluating finetuned model ...")
    finetuned_metrics = evaluate_model(finetuned_model, test_loader, device)
    finetuned_preds = collect_predictions(finetuned_model, test_loader, device)
    save_predictions_to_json(finetuned_preds, os.path.join(out_dir, "test_predictions_trained.json"))

    # Summary
    summary = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "test_file": test_file,
            "bert_path": bert_path,
            "ckpt_path": ckpt_path,
            "num_papers": num_papers,
            "out_dir": out_dir
        },
        "pretrained": baseline_metrics,
        "finetuned": finetuned_metrics,
        "delta": {
            "accuracy": finetuned_metrics["accuracy"] - baseline_metrics["accuracy"],
            "precision": finetuned_metrics["precision"] - baseline_metrics["precision"],
            "recall": finetuned_metrics["recall"] - baseline_metrics["recall"],
            "f1": finetuned_metrics["f1"] - baseline_metrics["f1"],
        }
    }
    out_json = os.path.join(out_dir, "test_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Pretty print
    print("\n========== Test Evaluation Summary ==========")
    print(f"Test file: {test_file}")
    print(f"Results saved to: {out_json}")
    print("\n-- Pretrained (un-finetuned) --")
    print(f"Acc: {baseline_metrics['accuracy']:.4f} | Prec: {baseline_metrics['precision']:.4f} | "
          f"Recall: {baseline_metrics['recall']:.4f} | F1: {baseline_metrics['f1']:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:", baseline_metrics["confusion_matrix"])

    print("\n-- Finetuned (trained) --")
    print(f"Acc: {finetuned_metrics['accuracy']:.4f} | Prec: {finetuned_metrics['precision']:.4f} | "
          f"Recall: {finetuned_metrics['recall']:.4f} | F1: {finetuned_metrics['f1']:.4f}")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:", finetuned_metrics["confusion_matrix"])

    print("\n-- Delta (finetuned - pretrained) --")
    for k, v in summary["delta"].items():
        print(f"{k}: {v:+.4f}")

    return out_dir


# =========================
# CLI
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Contrast eval: pretrained vs finetuned on test set")
    parser.add_argument('--bert_path', default=DEFAULT_BERT_PATH, help="Path to BERT (hf folder)")
    parser.add_argument('--test_file', default=DEFAULT_TEST_FILE, help="Path to test_dataset.json")
    parser.add_argument('--ckpt_path', default='', help="Path to finetuned best_model.pth")
    parser.add_argument('--date_str', default=os.environ.get("DATE_STR", date_str),
                        help="Optional fallback subdir under output/fuxian/")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    out_dir = evaluate_pre_and_post(
        bert_path=args.bert_path,
        test_file=args.test_file,
        ckpt_path=args.ckpt_path.strip(),
        batch_size=args.batch_size,
        out_dir_base=None
    )
    print(f"\n[Done] Outputs in: {out_dir}")
