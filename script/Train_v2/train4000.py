import os
import json
import glob
import random
from typing import List, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 关掉 tokenizers 并行的烦人 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 路径配置 =================
DATA_DIR = "/home/kzlab/muse/Savvy/Data_collection/main/process_dataset/train"
MODEL_DIR = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"
OUTPUT_DIR = "./bert_has_dataset_4000_out"

TARGET_TRAIN_SIZE = 4000  # 固定训练用 4000 条
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01


# ================= 工具函数 =================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_files(data_dir: str) -> List[Dict]:
    """
    把 data_dir 下所有 *_train_dataset.json 合并成一个 list
    """
    pattern = os.path.join(data_dir, "*_train_dataset.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No json files matched: {pattern}")

    all_samples: List[Dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容 {"data": [...]} 的情况
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                raise ValueError(f"Unexpected JSON format in {fp}")

        if not isinstance(data, list):
            raise ValueError(f"File {fp} is not a list of samples.")

        all_samples.extend(data)
        print(f"Loaded {len(data):5d} samples from {os.path.basename(fp)}")

    print(f"Total samples: {len(all_samples)}")
    return all_samples


def build_text_and_label(samples: List[Dict], label_key: str = "label"):
    """
    默认逻辑：
    - text = title + " [SEP] " + abstract  （如果两个字段都在）
    - 否则，如果有 "text" 字段就用 text
    - label 从 label_key 或 has_dataset 取
    """
    texts = []
    labels = []

    for s in samples:
        # 文本
        if "title" in s and "abstract" in s:
            text = f"{s['title']} [SEP] {s['abstract']}"
        elif "text" in s:
            text = str(s["text"])
        else:
            raise ValueError(f"Cannot find text field in sample keys: {list(s.keys())}")

        # 标签
        if label_key in s:
            y = int(s[label_key])
        elif "has_dataset" in s:
            y = int(s["has_dataset"])
        else:
            raise ValueError(
                f"Cannot find label field ('{label_key}' or 'has_dataset') "
                f"in sample keys: {list(s.keys())}"
            )

        texts.append(text)
        labels.append(y)

    return texts, labels


# ================= Dataset 定义 =================
class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # 简单起见用定长 padding
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ================= 训练 & 评估循环 =================
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].numpy()
            labels_all.extend(labels)

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=-1)
            preds_all.extend(preds)

    acc = accuracy_score(labels_all, preds_all)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, preds_all, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ================= 主函数 =================
def main():
    set_seed(42)

    # ----- 数据 -----
    all_samples = load_json_files(DATA_DIR)
    texts, labels = build_text_and_label(all_samples, label_key="label")
    n_total = len(texts)
    print(f"\nTotal samples: {n_total}")

    if n_total <= TARGET_TRAIN_SIZE:
        print(
            f"WARNING: total samples ({n_total}) <= {TARGET_TRAIN_SIZE}, "
            "use 90% train / 10% eval."
        )
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            texts,
            labels,
            test_size=0.1,
            random_state=42,
            stratify=labels if len(set(labels)) > 1 else None,
        )
    else:
        indices = np.arange(n_total)
        train_indices, eval_indices = train_test_split(
            indices,
            train_size=TARGET_TRAIN_SIZE,
            random_state=42,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        eval_texts = [texts[i] for i in eval_indices]
        eval_labels = [labels[i] for i in eval_indices]

    print(f"Train size: {len(train_texts)} (target {TARGET_TRAIN_SIZE})")
    print(f"Eval  size: {len(eval_texts)}\n")

    # ----- tokenizer & Dataset -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    train_dataset = ArxivDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    eval_dataset = ArxivDataset(eval_texts, eval_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----- 模型 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=num_labels
    )
    model.to(device)

    # ----- 优化器 -----
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    # ----- 训练循环 -----
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        metrics = evaluate(model, eval_loader, device)
        print(
            f"Eval accuracy: {metrics['accuracy']:.4f}, "
            f"precision: {metrics['precision']:.4f}, "
            f"recall: {metrics['recall']:.4f}, "
            f"f1: {metrics['f1']:.4f}"
        )

    # 最终再评估一次
    final_metrics = evaluate(model, eval_loader, device)
    print("\n== Final eval metrics (only 4000 train samples) ==")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    # ----- 保存模型 -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel & tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
