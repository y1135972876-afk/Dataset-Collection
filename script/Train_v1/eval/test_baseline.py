import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
from torch import nn

# 与训练代码完全一致的模型定义
class BertForSentenceClassification(nn.Module):
    def __init__(self, config, tokenizer=None, model_path=None):
        super().__init__()
        # 加载预训练的BERT模型
        if model_path:
            self.bert = BertModel.from_pretrained(model_path, config=config)
        else:
            self.bert = BertModel(config)

        self.tokenizer = tokenizer

        # 句子表示增强层
        self.sentence_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1),
            nn.GELU()
        )
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True  # 注意：后面会送入 [B, S, H]
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        device = input_ids.device
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]

        batch_size = input_ids.size(0)
        max_sentences = labels.size(1) if labels is not None else 1

        logits_list = []

        for i in range(batch_size):
            # 找到句子边界（真实 sep_token_id）
            sep_positions = (input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            cls_position = torch.tensor([0], device=device)  # CLS 位置
            # 将 CLS 位置加入边界
            all_positions = torch.cat([cls_position, sep_positions])

            sentence_representations = []
            # 基于 [CLS] 到每个 [SEP] 切分句子
            for j in range(len(all_positions) - 1):
                start_pos = all_positions[j]
                end_pos = all_positions[j + 1]  # 包含 [SEP]

                # [start, end] 的 token 表示
                sentence_tokens = sequence_output[i, start_pos:end_pos + 1]  # [L, H]
                sentence_mask = attention_mask[i, start_pos:end_pos + 1].unsqueeze(-1)  # [L, 1]

                # 平均池化（masked mean）
                masked_tokens = sentence_tokens * sentence_mask
                denom = sentence_mask.sum() + 1e-10
                sentence_rep = masked_tokens.sum(dim=0) / denom  # [H]

                # 句子级编码
                sentence_rep = self.sentence_encoder(sentence_rep)  # [H]
                sentence_representations.append(sentence_rep)

            if sentence_representations:
                # [num_sentences, H]
                sentence_tensor = torch.stack(sentence_representations, dim=0)
                current_sentences = sentence_tensor.size(0)

                # --- 修复点：MultiheadAttention 需要 [B, S, H] ---
                sentence_tensor_b = sentence_tensor.unsqueeze(0)  # [1, S, H]
                attended_b, _ = self.sentence_attention(
                    sentence_tensor_b, sentence_tensor_b, sentence_tensor_b
                )
                attended = attended_b.squeeze(0)  # [S, H]

                # 只对真实句子过分类器
                real_logits = self.classifier(attended)  # [S, 2]

                # 对齐到 max_sentences：不足的部分用 0 填充（不会参与 loss）
                if current_sentences < max_sentences:
                    pad = torch.zeros((max_sentences - current_sentences, 2), device=device)
                    sentence_logits = torch.cat([real_logits, pad], dim=0)  # [max_sentences, 2]
                else:
                    sentence_logits = real_logits[:max_sentences]
                logits_list.append(sentence_logits)
            else:
                # 没有句子时的兜底（全 0 logits）
                empty_logits = torch.zeros(max_sentences, 2, device=device)
                logits_list.append(empty_logits)

        # [B, max_sentences, 2]
        padded_logits = torch.stack(logits_list, dim=0)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_logits = padded_logits.view(-1, 2)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

        return loss, padded_logits


# 数据集类
class PaperDataset(Dataset):
    def __init__(self, papers, tokenizer, max_length=512, min_sentences=3, context_size=2):
        print("\nInitializing PaperDataset...")
        print(f"Number of papers to process: {len(papers)}")
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_sentences = min_sentences
        self.context_size = context_size
        self.flat_examples = self._prepare_examples()

    def _get_window_size(self, sentences, start_idx):
        """基于token数量确定窗口大小"""
        current_tokens = 0
        window_size = 0
        for i in range(start_idx, len(sentences)):
            # 以不加特殊符号的方式估 token 数
            tokens = self.tokenizer.encode(sentences[i]['text'], add_special_tokens=False)
            # 句与句之间还会插入一个 SEP；序列开头还有 CLS
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
                i += step_size
        return examples

    def __len__(self):
        return len(self.flat_examples)

    def __getitem__(self, idx):
        example = self.flat_examples[idx]
        sentences = example['sentences']

        # 文本与标签
        text_list = [sent['text'] for sent in sentences]
        labels = [sent['label'] for sent in sentences]

        # === 关键修复：用真实 token 序列拼接 CLS/SEP ===
        sent_token_ids = []
        for s in text_list:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            sent_token_ids.append(ids)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        # [CLS] s1 [SEP] s2 [SEP] ... sN [SEP]
        input_ids_list = [cls_id]
        for ids in sent_token_ids:
            input_ids_list.extend(ids)
            input_ids_list.append(sep_id)

        # 截断到 max_length，保证最后一个位置是 SEP
        if len(input_ids_list) > self.max_length:
            input_ids_list = input_ids_list[:self.max_length]
            if input_ids_list[-1] != sep_id:
                input_ids_list[-1] = sep_id

        attention_mask_list = [1] * len(input_ids_list)

        # padding
        pad_len = self.max_length - len(input_ids_list)
        if pad_len > 0:
            input_ids_list += [self.tokenizer.pad_token_id] * pad_len
            attention_mask_list += [0] * pad_len

        encodings = {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
        }

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.tensor(example['indices'], dtype=torch.long),
            'paper_name': example['paper_name'],
            'has_positive': example['has_positive'],
            'sentences': text_list
        }


def custom_collate_fn(batch):
    input_ids = []
    attention_masks = []
    labels = []
    indices = []
    paper_names = []
    sentences_list = []

    max_sentences = max(len(item['labels']) for item in batch)

    for item in batch:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])

        # 填充标签到相同长度（-100 不参与损失/评估）
        curr_labels = item['labels']
        if len(curr_labels) < max_sentences:
            padding = torch.full((max_sentences - len(curr_labels),), -100, dtype=curr_labels.dtype)
            curr_labels = torch.cat([curr_labels, padding])
        labels.append(curr_labels)

        curr_indices = item['indices']
        if len(curr_indices) < max_sentences:
            padding = torch.full((max_sentences - len(curr_indices),), -1, dtype=curr_indices.dtype)
            curr_indices = torch.cat([curr_indices, padding])
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


def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    data = raw_data['data']
    papers = []
    current_paper = None

    for item in sorted(data, key=lambda x: (x['paper_name'], x['section'], x['paragraph_id'])):
        if current_paper is None or item['paper_name'] != current_paper['paper_name']:
            if current_paper is not None:
                papers.append(current_paper)
            current_paper = {
                'paper_name': item['paper_name'],
                'sentences': []
            }
        current_paper['sentences'].extend(item['sentences'])

    if current_paper is not None:
        papers.append(current_paper)

    return papers


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_sentences = []
    all_paper_names = []
    all_confidences = []

    nan_warned = False  # 只提示一次 NaN

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs[1]  # [batch_size, max_sentences, 2]

            # --- 兜底：softmax 前清理 NaN/Inf ---
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).float()

            # 获取预测结果
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)

            if torch.isnan(confidences).any() and not nan_warned:
                tqdm.write("Warning: NaN values in confidences detected (showing once)")
                nan_warned = True
            confidences = torch.nan_to_num(confidences, nan=0.5, posinf=1.0, neginf=0.0)

            # 处理每个样本
            for i in range(len(batch['paper_names'])):
                paper_name = batch['paper_names'][i]
                sentences = batch['sentences'][i]
                paper_labels = labels[i]
                paper_predictions = predictions[i]
                paper_confidences = confidences[i]

                # 只处理有效标签（非填充）
                valid_mask = paper_labels != -100
                valid_indices = torch.arange(len(paper_labels), device=device)[valid_mask]

                for idx in valid_indices:
                    idx_item = idx.item()
                    if idx_item < len(sentences):  # 确保索引在句子列表范围内
                        all_predictions.append(paper_predictions[idx_item].item())
                        all_labels.append(paper_labels[idx_item].item())
                        all_sentences.append(sentences[idx_item])
                        all_paper_names.append(paper_name)
                        all_confidences.append(paper_confidences[idx_item].item())

    return all_predictions, all_labels, all_sentences, all_paper_names, all_confidences


def calculate_metrics(predictions, labels):
    """计算详细的评估指标"""
    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = np.mean(predictions == labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    class_report = classification_report(
        labels, predictions, target_names=['Negative', 'Positive'], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_bar(metrics, save_path):
    """绘制指标条形图"""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')

    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences, labels, predictions, save_path):
    """绘制置信度分布"""
    confidences = np.array(confidences)
    labels = np.array(labels)
    predictions = np.array(predictions)

    if np.isnan(confidences).any():
        print("Warning: NaN values found in confidences, replacing with 0.5")
        confidences = np.nan_to_num(confidences, nan=0.5, posinf=1.0, neginf=0.0)

    conf_neg = confidences[labels == 0]
    conf_pos = confidences[labels == 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    if len(conf_neg) > 0:
        plt.hist(conf_neg, bins=20, alpha=0.7, color='red', label='Negative')
    if len(conf_pos) > 0:
        plt.hist(conf_pos, bins=20, alpha=0.7, color='blue', label='Positive')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    correct_mask = (predictions == labels)
    conf_correct = confidences[correct_mask]
    conf_incorrect = confidences[~correct_mask]

    if len(conf_correct) > 0:
        plt.hist(conf_correct, bins=20, alpha=0.7, color='green', label='Correct')
    if len(conf_incorrect) > 0:
        plt.hist(conf_incorrect, bins=20, alpha=0.7, color='orange', label='Incorrect')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Prediction Correctness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(predictions, labels, sentences, paper_names, confidences, metrics, save_path):
    """保存详细的结果分析"""
    if 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], np.ndarray):
        metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()

    results = {
        'metadata': {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(predictions),
            'positive_samples': int(np.sum(labels)),
            'negative_samples': int(len(labels) - np.sum(labels)),
            'overall_metrics': metrics
        },
        'detailed_predictions': []
    }

    # 按论文组织
    paper_results = {}
    for i, paper_name in enumerate(paper_names):
        if paper_name not in paper_results:
            paper_results[paper_name] = []
        paper_results[paper_name].append({
            'sentence': sentences[i],
            'true_label': int(labels[i]),
            'predicted_label': int(predictions[i]),
            'confidence': float(confidences[i]),
            'correct': int(predictions[i] == labels[i])
        })
    results['paper_wise_results'] = paper_results

    # 每篇论文准确率
    paper_metrics = {}
    for paper_name, sentences_data in paper_results.items():
        correct = sum(1 for s in sentences_data if s['correct'])
        total = len(sentences_data)
        paper_metrics[paper_name] = {
            'accuracy': correct / total if total > 0 else 0.0,
            'total_sentences': total,
            'correct_predictions': correct
        }
    results['paper_metrics'] = paper_metrics

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    # 路径配置（按需修改）
    model_path = "/home/kzlab/muse/Savvy/Data_collection/output/fuxian/1224/best_model.pth"
    test_data_path = "/home/kzlab/muse/Savvy/Data_collection/dataset/test_dataset.json"
    bert_model_path = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"
    output_dir = "/home/kzlab/muse/Savvy/Data_collection/output/evaluation_results"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("Loading tokenizer and model configuration...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    config = BertConfig.from_pretrained(bert_model_path, num_labels=2)

    print("Initializing model with the same architecture as training...")
    model = BertForSentenceClassification(config, tokenizer=tokenizer, model_path=bert_model_path)

    print(f"Loading model from: {model_path}")
    # 注意：某些 PyTorch 版本需要 weights_only=False
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("Loading test data...")
    test_papers = load_data(test_data_path)
    print(f"Loaded {len(test_papers)} test papers")

    test_dataset = PaperDataset(
        papers=test_papers,
        tokenizer=tokenizer,
        max_length=512,
        min_sentences=3,
        context_size=2
    )
    print(f"Test dataset contains {len(test_dataset)} windows")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # 便于逐句分析
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print("Starting evaluation...")
    predictions, labels, sentences, paper_names, confidences = evaluate_model(
        model, test_dataloader, device
    )

    if len(predictions) == 0:
        print("Warning: No predictions were generated. Check the data processing.")
        return

    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, labels)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Total samples: {len(predictions)}")
    print(f"Positive samples: {int(np.sum(labels))}")
    print(f"Negative samples: {int(len(labels) - np.sum(labels))}")

    print("Generating visualizations...")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)

    metrics_path = os.path.join(output_dir, "metrics_bar.png")
    plot_metrics_bar(metrics, metrics_path)

    conf_path = os.path.join(output_dir, "confidence_distribution.png")
    plot_confidence_distribution(confidences, labels, predictions, conf_path)

    results_path = os.path.join(output_dir, "detailed_results.json")
    detailed_results = save_detailed_results(
        predictions, labels, sentences, paper_names, confidences, metrics, results_path
    )

    print("\nPaper-wise Accuracy:")
    for paper_name, paper_metric in detailed_results['paper_metrics'].items():
        print(f"{paper_name}: {paper_metric['accuracy']:.4f} "
              f"({paper_metric['correct_predictions']}/{paper_metric['total_sentences']})")

    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=['Negative', 'Positive'], zero_division=0))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
