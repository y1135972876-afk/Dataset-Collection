"""
utils.py — 通用训练/评估工具库
支持：BERT / Longformer / BigBird（以及其它 AutoModel 兼容模型）
目标：
- 一套 Dataset / DataLoader / 训练与评估 / 存储与可视化 的公用函数
- 可自由切换模型、epoch、数据集、batch size、最大长度等
- 统一句子级分类（将长文切分为句子窗口，在序列内部按 SEP 分句并逐句分类）

使用示例（train.py 里）：
    from utils import (
        set_seed, build_tokenizer, build_backbone_model,
        SentenceClassifier, load_json_papers,
        make_dataloader, train_one_epoch, evaluate,
        save_predictions_to_json, plot_history, ensure_dir,
    )

    tokenizer = build_tokenizer(model_name_or_path)
    backbone  = build_backbone_model(model_name_or_path)  # 返回 AutoModel
    model     = SentenceClassifier(backbone.config, backbone, tokenizer, num_labels=2).to(device)

    train_loader = make_dataloader(train_papers, tokenizer, batch_size=1, max_length=tokenizer.model_max_length)
    val_loader   = make_dataloader(val_papers, tokenizer,   batch_size=1, max_length=tokenizer.model_max_length, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    best = {"acc": 0.0}
    history = {k: [] for k in ["train_acc","train_prec","train_recall","train_f1","val_acc","val_prec","val_recall","val_f1"]}
    for epoch in range(num_epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, grad_accum_steps=4)
        va, preds = evaluate(model, val_loader, device)
        for k in ["acc","prec","recall","f1"]:
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])
        if va["acc"] > best["acc"]:
            best = va
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        plot_history(history, os.path.join(save_dir, "plots"), epoch+1)
        save_predictions_to_json(preds, os.path.join(save_dir, "predictions", f"preds_epoch_{epoch+1}.json"))
"""

from __future__ import annotations
import os
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


# ------------------------- 基础工具 -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ------------------------- 数据加载与预处理 -------------------------

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


class SentenceWindowDataset(Dataset):
    """将论文的句子按最大 token 长度切窗口；窗口内部用 tokenizer.sep_token 连接。
    支持 min_sentences、context_size（若窗口内含正样本，会收缩到正样本附近的上下文）。
    """
    def __init__(
        self,
        papers: List[Dict],
        tokenizer,
        max_length: int,
        min_sentences: int = 3,
        context_size: int = 2,
    ) -> None:
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_sentences = min_sentences
        self.context_size = context_size
        self.sep_token = tokenizer.sep_token or "</s>"  # 兼容 roberta/longformer/bigbird/bert
        self._examples = self._build_examples()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self._examples[idx]
        texts = [s['text'] for s in ex['sentences']]
        labels = [s['label'] for s in ex['sentences']]
        full_text = f" {self.sep_token} ".join(texts)
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
            'indices': torch.tensor(ex['indices'], dtype=torch.long),
            'paper_name': ex['paper_name'],
            'sentences': texts,
        }

    # --------- 内部：根据 token 长度滚动生成窗口 ---------
    def _get_window_size(self, sentences: List[Dict], start: int) -> int:
        cur_tokens = 0
        win = 0
        for i in range(start, len(sentences)):
            tokens = self.tokenizer.encode(sentences[i]['text'], add_special_tokens=False)
            next_len = cur_tokens + len(tokens) + (1 if cur_tokens > 0 else 2)  # 中间 sep + 首尾 special
            if next_len >= self.max_length - 2:
                break
            cur_tokens = next_len
            win += 1
        return max(self.min_sentences, win)

    def _build_examples(self) -> List[Dict]:
        examples = []
        
        for paper in self.papers:
            sentences = paper['sentences']
            i = 0
            
            while i < len(sentences):
                # 1. 获取当前位置的窗口大小
                window_size = self._get_window_size(sentences, i)
                
                # 2. 提取当前窗口的句子
                end_idx = min(i + window_size, len(sentences)) #保证末尾end_idx不超过句子总数len(sentences)
                current_window = sentences[i:end_idx] # 当前窗口内容
                current_indices = list(range(i, end_idx)) # 当前窗口id
                
                # 3. 计算窗口内的token数量
                window_tokens = sum(len(self.tokenizer.encode(sent['text'], add_special_tokens=False))
                                  for sent in current_window)
                
                # 4. 检查窗口中是否包含正样本
                has_positive = any(sent['label'] == 1 for sent in current_window)
                
                # 5. 如果包含正样本，确保有足够的上下文
                if has_positive:
                    # 找到第一个正样本的位置
                    first_positive = next(idx for idx, sent in enumerate(current_window) 
                                       if sent['label'] == 1)
                    # 确保正样本前后有足够的上下文
                    start_offset = max(0, first_positive - self.context_size) #在当前窗口内尽量为第一个正样本保留前后 context_size 个句子的上下文。”
                    end_offset = min(len(current_window), 
                                   first_positive + self.context_size + 1)
                    # 调整窗口范围
                    current_window = current_window[start_offset:end_offset]
                    current_indices = current_indices[start_offset:end_offset]
                    window_tokens = sum(len(self.tokenizer.encode(sent['text'], 
                                      add_special_tokens=False))
                                      for sent in current_window)
                
                # 6. 将处理后的窗口添加到示例中
                if len(current_window) >= self.min_sentences:
                    examples.append({
                        'sentences': current_window,
                        'indices': current_indices,
                        'paper_name': paper['paper_name'],
                        'window_tokens': window_tokens,
                        'has_positive': has_positive
                    })
                
                # 7. 动态调整步长
                # 如果窗口包含正样本，使用较小的步长以避免遗漏  ??? 此处是不是有问题
                # step_size = max(1, window_size // 2) if has_positive else window_size // 2
                step_size = max(1, window_size // 3) if has_positive else window_size // 2
                i += step_size
        
        return examples


def collate_sentence_windows(batch: List[Dict]) -> Dict:
    input_ids = [b['input_ids'] for b in batch]
    attn = [b['attention_mask'] for b in batch]
    labels = [b['labels'] for b in batch]
    idxs   = [b['indices'] for b in batch]
    names  = [b['paper_name'] for b in batch]
    texts  = [b['sentences'] for b in batch]

    max_sent = max(len(lb) for lb in labels)
    pad_labels, pad_indices = [], []
    for lb, idc in zip(labels, idxs):
        if len(lb) < max_sent:
            pad = torch.full((max_sent - len(lb),), -100, dtype=lb.dtype)
            lb = torch.cat([lb, pad])
        if len(idc) < max_sent:
            pad = torch.full((max_sent - len(idc),), -1, dtype=idc.dtype)
            idc = torch.cat([idc, pad])
        pad_labels.append(lb)
        pad_indices.append(idc)

    batch_out = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attn),
        'labels': torch.stack(pad_labels),
        'indices': torch.stack(pad_indices),
        'paper_names': names,
        'sentences': texts,
    }
    return batch_out


def make_dataloader(
    papers: List[Dict],
    tokenizer,
    batch_size: int = 1,
    max_length: Optional[int] = None,
    min_sentences: int = 3,
    context_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if max_length is None:
        # 使用 tokenizer.model_max_length；若超大值(1e30)，回退到 4096
        m = tokenizer.model_max_length
        max_length = 4096 if m > 1e6 else int(m)
    ds = SentenceWindowDataset(papers, tokenizer, max_length, min_sentences, context_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sentence_windows, # ?
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ------------------------- 模型构建 -------------------------

def build_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def build_backbone_model(model_name_or_path: str):
    """返回 AutoModel（非 *ForSequenceClassification*），用于统一自定义句子级头。"""
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    return AutoModel.from_pretrained(model_name_or_path, config=cfg)


class SentenceClassifier(nn.Module):
    """通用句子级分类头（与训练/推理数据构造保持一致）。

    思路小结
    -------
    1) 上游把一段文本窗口里的多句，用 tokenizer.sep_token (“[SEP]”、"</s>" 等) 连接成“单个序列”送入 backbone（BERT/Roberta/Longformer/BigBird 任一 AutoModel）。
    2) 在 forward 里，先用 SEP 的位置把这一整段序列切回「句子片段」；
       对每个句子片段的 token 表示做一次 **平均池化(average pooling)** 得到句向量。
    3) 每个句向量过一个两层 MLP（带 LayerNorm/Dropout/GELU）输出 2 类 logits（是否“数据集描述”）。
    4) 同一 batch 中每个样本的句子数不一样，因此：
       - 训练时上游会把标签 `labels` 按「句子数」对齐，并用 `-100` 作为 padding（忽略项）。
       - 这里也把句向量对齐到统一长度 `max_sentences`（与 `labels.size(1)` 一致），多余的位置用 0 向量补齐。
    5) 返回 `(loss, logits)`，其中 `logits` 形状为 `[B, max_sentences, num_labels]`。
    """

    def __init__(self, config, backbone: nn.Module, tokenizer, num_labels: int = 2):
        super().__init__()
        self.config = config                 # HF 的模型配置（含 hidden_size 等）
        self.backbone = backbone             # 任意 AutoModel（不带分类头）
        self.tokenizer = tokenizer           # 为了拿到 sep_token_id
        self.num_labels = num_labels

        hidden = config.hidden_size
        # 句向量 -> 分类 的小头（两层 MLP + LN + Dropout + GELU）
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(hidden, num_labels),
        )
        # 初始化 Linear 的权重/偏置（与 HF 初始化风格一致）
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=getattr(config, 'initializer_range', 0.02))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        参数
        ----
        input_ids:      [B, L]  上游把一个「窗口」内的多句拼成一个序列（句间以 SEP 分隔）
        attention_mask: [B, L]  padding 位置为 0
        labels:         [B, S]  句级标签（S 是该 batch 对齐后的“句子槽位数”），无标签位置为 -100

        返回
        ----
        loss:   标量（若传入 labels，则计算交叉熵；否则为 None）
        logits: [B, S, C]  C=num_labels=2，句级 logits
        """
        # 1) 过 backbone，取最后一层的 token 表示
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs[0]  # [B, L, H]  注意 AutoModel.__call__ 输出的第一个是 last_hidden_state
        B, L, H = seq.shape

        # 2) 确定本 batch 要对齐到的“句子槽位数” S：
        #    - 若有 labels，就直接用 labels.size(1)（和上游对齐一致）
        #    - 否则根据 SEP 数目估计（句子数 ≈ SEP 个数 + 1）
        max_sentences = labels.size(1) if labels is not None else self._infer_max_sentences_from_seps(input_ids)

        logits_list = []
        sep_id = self.tokenizer.sep_token_id    # SEP 的 token id（BERT 是 102；Roberta 是 2）
        device = input_ids.device

        # 3) 对每个样本 i，按 SEP 把整段序列切分回“句片段”
        for i in range(B):
            # 找到该样本中所有 SEP 的位置（1D 索引张量）
            sep_pos = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
            # 把 CLS(索引 0) 也视作一个“边界起点”，与每个 SEP 一起构成切分边界
            cls_pos = torch.tensor([0], device=device)  # 假设位置 0 是 [CLS]
            bounds = torch.cat([cls_pos, sep_pos])      # 长度大约等于“句子数”

            sent_reps = []  # 存放每个句子的 [H] 向量
            # 逐段构造：每段为 [bounds[j], bounds[j+1]]（含 SEP）
            for j in range(len(bounds) - 1): #例如bounds： tensor([  0,  18,  33,  62,  88, 108], device='cuda:0')，这里是一个一个句子处理，每个句子里有多个token，每个token是一维度，这里是id
                st = bounds[j]
                ed = bounds[j + 1]
                # 取该句的 token 表示：seq[i, st:ed+1] -> [T, H]
                tok = seq[i, st:ed+1] # 包含 SEP
                # 取对应的 mask 并扩展维度：[T] -> [T,1]
                msk = attention_mask[i, st:ed+1].unsqueeze(-1)
                # 做“带 mask 的求和 / 归一化”即 **平均池化**，得到句向量 [H]
                masked = tok * msk                          # padding 位置清零
                denom = msk.sum().clamp(min=1.0)            # 防止除 0
                rep = masked.sum(dim=0) / denom             # [H]，句向量
                sent_reps.append(rep)

            if len(sent_reps) == 0:
                # 兜底：万一没找到 SEP（极端截断/异常输入），就用 CLS 表示一个“句子”
                rep = seq[i, 0]                             # [H]
                sent_reps = [rep]

            # 将变长的句向量列表堆叠为 [S_i, H]
            S_i = len(sent_reps)
            reps = torch.stack(sent_reps)                   # [S_i, H]
            # 与 batch 内的对齐 S (=max_sentences) 对齐：不足补零，过长截断
            if S_i < max_sentences: #！ 这里确实大小不一致
                pad = torch.zeros(max_sentences - S_i, H, device=device)
                reps = torch.cat([reps, pad], dim=0)        # [S, H]
            else:
                reps = reps[:max_sentences]                 # [S, H]

            # 4) 句向量过 MLP 分类头 -> 得到 [S, C] 的句级 logits
            logit = self.classifier(reps)                   # [S, num_labels]
            logits_list.append(logit)

        # 5) 拼回 batch 维度tokenizer
        logits = torch.stack(logits_list, dim=0)            # [B, S, C]

        # 6) 若给了 labels，则计算交叉熵：
        #    - labels 展平成 1D，并与 logits.view(-1, C) 对齐；
        #    - 忽略值 -100 会被 CrossEntropyLoss 丢弃（与上游 collate 的对齐策略配套）。
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def _infer_max_sentences_from_seps(self, input_ids: torch.Tensor) -> int:
        """
        在没提供 labels 对齐长度时，估算“需要对齐到的句子数”：
        近似用“序列中的 SEP 个数 + 1”（因为句间由 SEP 分隔）。
        注意：若上游截断导致末尾 SEP 被截掉，这个估计会偏小；但对无标签推理场景够用。
        """
        sep_id = self.tokenizer.sep_token_id
        if sep_id is None:
            return 1
        # 统计每个样本的 SEP 个数（[B]），取最大值并 +1
        sep_counts = (input_ids == sep_id).sum(dim=1)  # [B]
        return int(sep_counts.max().item() )




# ------------------------- 指标/可视化/预测保存 -------------------------

def compute_metrics(pred: torch.Tensor, gold: torch.Tensor) -> Dict[str, float]:
    """忽略 -100 标签，返回 acc/prec/recall/f1。输入均为一维或可展平。"""
    mask = gold != -100
    pred = pred[mask]
    gold = gold[mask]
    if pred.numel() == 0:
        return {"acc": 0.0, "prec": 0.0, "recall": 0.0, "f1": 0.0}
    # sklearn 指标：在 CPU 上计算
    p_np = pred.detach().cpu().numpy()
    g_np = gold.detach().cpu().numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(g_np, p_np, average='binary', zero_division=0)
    acc = (pred == gold).float().mean().item()
    return {"acc": acc, "prec": float(prec), "recall": float(rec), "f1": float(f1)}


def plot_history(history: Dict[str, List[float]], save_dir: str, epoch: int) -> None:
    ensure_dir(save_dir)
    plt.figure(figsize=(12, 8))
    items = [("acc","Accuracy"),("prec","Precision"),("recall","Recall"),("f1","F1")]
    for i,(k,title) in enumerate(items,1):
        plt.subplot(2,2,i)
        plt.plot(history.get(f"train_{k}",[]), label='Train', marker='o')
        plt.plot(history.get(f"val_{k}",[]),   label='Val',   marker='o')
        plt.title(title); plt.xlabel('Epoch'); plt.ylabel(title); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_epoch_{epoch}.png'))
    plt.close()


def collect_predictions(model, dataloader, device):
    model.eval()
    predictions_data = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Collecting predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            indices = batch['indices'] #来自 Dataset.__getitem__ 的 ex['indices']（窗口内真实句子数 = N）
            paper_names = batch['paper_names']
            sentences = batch['sentences']  # 这里包含完整的句子文本
            
            #debug一下，可以删除
            sep_id = model.tokenizer.sep_token_id
            last_idx = attention_mask.sum(dim=1) - 1
            last_tok = input_ids[torch.arange(input_ids.size(0)), last_idx]
            print("末尾是否SEP：", (last_tok == sep_id).tolist())
            #debug一下    
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 获取预测概率和标签
            logits = outputs[1]  # [batch_size, max_sentences, 2]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 遍历每个batch中的样本
            for b in range(len(paper_names)):
                # 获取有效的句子索引（排除填充的-1）
                valid_indices = indices[b][indices[b] != -1]
                valid_predictions = predictions[b][:len(valid_indices)]
                valid_probs = probs[b][:len(valid_indices)]
                valid_labels = labels[b][:len(valid_indices)]
                valid_sentences = sentences[b][:len(valid_indices)]  # 获取完整的句子内容
                
                # 收集每个句子的预测信息
                sentences_data = []
                for idx, pred, prob, label, sent_text in zip(
                    valid_indices, 
                    valid_predictions, 
                    valid_probs,
                    valid_labels,
                    valid_sentences
                ):
                    if label != -100:  # 排除填充的标签
                        sentences_data.append({
                            'sentence_index': idx.item(),
                            'sentence_text': sent_text,  # 完整的句子文本
                            'predicted_label': pred.item(),
                            'true_label': label.item(),
                            'confidence': prob[pred].item(),
                            'probabilities': {
                                'class_0': prob[0].item(),
                                'class_1': prob[1].item()
                            }
                        })
                
                # 添加到预测数据中
                if sentences_data:  # 只添加有效的预测数据
                    predictions_data.append({
                        'paper_name': paper_names[b],
                        'sentences': sentences_data
                    })
    
    return predictions_data



def save_predictions_to_json(predictions: List[Dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    total = sum(len(p['sentences']) for p in predictions)
    correct = sum(1 for p in predictions for s in p['sentences'] if s['predicted_label'] == s['true_label'])
    pos = sum(1 for p in predictions for s in p['sentences'] if s['true_label'] == 1)
    meta = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'num_papers': len(predictions),
        'total_sentences': total,
        'statistics': {
            'accuracy': (correct / total) if total > 0 else 0.0,
            'total_correct': correct,
            'total_positive_samples': pos,
            'positive_ratio': (pos / total) if total > 0 else 0.0,
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': meta, 'predictions': predictions}, f, ensure_ascii=False, indent=2)


# ------------------------- 训练 / 验证 -------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,   # 梯度累积步数；>1 时等价于扩大会显存的“有效 batch size”
    use_amp: bool = False,       # 是否启用混合精度（自动混合精度 AMP），可减少显存/加速
) -> Dict[str, float]:
    # AMP 的梯度缩放器；开启 AMP 时有效，关闭时内部等价于空操作
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()                # 进入训练模式（启用 Dropout/BN 的训练分支等）
    running_loss = 0.0           # 累加平均 loss（按 step 取均值）
    all_pred, all_gold = [], []  # 收集所有 step 的预测与标签，用于最终计算指标

    # 迭代一个 epoch 内的所有 mini-batch
    for step, batch in enumerate(tqdm(dataloader, desc='Training')):
        # 将 batch 张量移到目标设备（GPU/CPU）
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # autocast：在前向中自动以半精度/单精度混合计算（仅在 use_amp=True 时生效）
        with torch.cuda.amp.autocast(enabled=use_amp):
            # 前向传播：模型返回 (loss, logits)
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            # 梯度累积：把当前 step 的 loss 缩小，以在累计 N 次 backward 后等价于一次大 batch
            loss = loss / grad_accum_steps

        # 用 scaler.scale 包装 loss.backward()，避免半精度下的梯度下溢
        scaler.scale(loss).backward()

        # 每累积 grad_accum_steps 次再进行一次优化器更新
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)           # 等价于 optimizer.step()，但带 AMP 的缩放/回退逻辑
            scaler.update()                  # 动态调整缩放因子
            optimizer.zero_grad(set_to_none=True)  # 清梯度；置 None 可减少显存、略微提速

        # 记录当前 step 的标量 loss（注意是已除以 grad_accum_steps 后的值）
        running_loss += loss.item()

        # 将本 step 的 logits 转为类别预测；展平到一维以便与 labels 对齐
        preds = torch.argmax(logits, dim=-1).reshape(-1)
        golds = labels.reshape(-1)

        # 只收集张量的“断开计算图”的拷贝，防止显存被计算图占用
        all_pred.append(preds.detach())
        all_gold.append(golds.detach())

    # 将所有 step 的预测/标签在第 0 维拼接，构成完整 epoch 的样本集
    if all_pred:
        all_pred = torch.cat(all_pred)
        all_gold = torch.cat(all_gold)
    else:
        # 边界情况：空 dataloader 时返回空张量，防止后续计算崩溃
        all_pred = torch.tensor([], device=device)
        all_gold = torch.tensor([], device=device)

    # 计算精度/查准/查全/F1（内部会忽略 -100 的填充值）
    metrics = compute_metrics(all_pred, all_gold)
    # 记录 epoch 平均 loss（按 step 数均值）
    metrics['loss'] = running_loss / max(1, len(dataloader))
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str,float], List[Dict]]:
    model.eval()
    val_loss = 0.0
    all_pred, all_gold = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).reshape(-1) # preds 为预测标签
            golds = labels.reshape(-1)  # golds 为预测标签
            all_pred.append(preds.detach())
            all_gold.append(golds.detach())

    if all_pred:
        all_pred = torch.cat(all_pred)
        all_gold = torch.cat(all_gold)
    else:
        all_pred = torch.tensor([], device=device)
        all_gold = torch.tensor([], device=device)

    metrics = compute_metrics(all_pred, all_gold)
    metrics['loss'] = val_loss / max(1, len(dataloader))

    # 同时收集一份结构化预测，便于调试/追踪
    preds_struct = collect_predictions(model, dataloader, device)
    return metrics, preds_struct


# ------------------------- 运行目录/命名辅助 -------------------------

def make_experiment_dir(root: str, model_name_or_path: str, dataset_tag: str, tag: Optional[str] = None) -> str:
    """根据模型名/数据集标签/时间戳生成保存目录。"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.basename(model_name_or_path.rstrip('/'))
    name = f"{model_tag}__{dataset_tag}__{ts}"
    if tag:
        name += f"__{tag}"
    out = os.path.join(root, name)
    ensure_dir(out)
    ensure_dir(os.path.join(out, 'predictions'))
    ensure_dir(os.path.join(out, 'plots'))
    return out
