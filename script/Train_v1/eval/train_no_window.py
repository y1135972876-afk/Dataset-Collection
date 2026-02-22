"""
去掉滑动窗口版：逐句独立分类
保持训练参数基本不变
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import os
from datetime import datetime

date_str = "1224"

def calculate_metrics(predictions, labels):
    """计算评估指标（逐句分类，无需忽略填充）"""
    if predictions.is_cuda:
        predictions_cpu = predictions.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
    else:
        predictions_cpu = predictions.numpy()
        labels_cpu = labels.numpy()

    if len(predictions_cpu) == 0 or len(labels_cpu) == 0:
        return 0.0, 0.0, 0.0, 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_cpu, predictions_cpu, average='binary', zero_division=0
    )
    accuracy = (predictions == labels).float().mean().item()
    return accuracy, precision, recall, f1

def plot_metrics(history, epoch):
    fig_save_path = f"/home/kzlab/muse/Savvy/Data_collection/output/fuxian/no_window/{date_str}"
    os.makedirs(fig_save_path, exist_ok=True)

    plt.figure(figsize=(15, 10))
    metrics = ['acc', 'prec', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 2, i)
        plt.plot(history[f'train_{metric}'], label='Train', marker='o')
        plt.plot(history[f'val_{metric}'], label='Validation', marker='o')
        plt.title(f'{title} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, f'metrics_epoch_{epoch}.png'))
    plt.close()

class PaperDataset(Dataset):
    """
    逐句样本：
    每条样本 = (sentence_text, label, paper_name, sentence_index)
    """
    def __init__(self, papers, tokenizer, max_length=512):
        print("\nInitializing PaperDataset (sentence-level)...")
        print(f"Number of papers to process: {len(papers)}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._flatten(papers)

    def _flatten(self, papers):
        samples = []
        for paper in papers:
            name = paper['paper_name']
            for idx, sent in enumerate(paper['sentences']):
                text = sent['text']
                label = sent['label']
                samples.append({
                    'paper_name': name,
                    'sentence_index': idx,
                    'text': text,
                    'label': label
                })
        print(f"Total sentence samples: {len(samples)}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'paper_name': item['paper_name'],
            'sentence_index': item['sentence_index'],
            'sentence_text': item['text']
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])

    paper_names = [b['paper_name'] for b in batch]
    sent_indices = [b['sentence_index'] for b in batch]
    sent_texts = [b['sentence_text'] for b in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'paper_names': paper_names,
        'sentence_indices': sent_indices,
        'sentences': sent_texts
    }

def collect_predictions(model, dataloader, device):
    model.eval()
    predictions_data_by_paper = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Collecting predictions'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits  # [B, 2]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            for i in range(input_ids.size(0)):
                paper = batch['paper_names'][i]
                if paper not in predictions_data_by_paper:
                    predictions_data_by_paper[paper] = {'paper_name': paper, 'sentences': []}
                predictions_data_by_paper[paper]['sentences'].append({
                    'sentence_index': batch['sentence_indices'][i],
                    'sentence_text': batch['sentences'][i],
                    'predicted_label': preds[i].item(),
                    'true_label': labels[i].item(),
                    'confidence': probs[i, preds[i]].item(),
                    'probabilities': {
                        'class_0': probs[i, 0].item(),
                        'class_1': probs[i, 1].item()
                    }
                })

    return list(predictions_data_by_paper.values())

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
            current_paper = {'paper_name': item['paper_name'], 'sentences': []}
        current_paper['sentences'].extend(item['sentences'])

    if current_paper is not None:
        papers.append(current_paper)
    return papers

def save_predictions_to_json(predictions_data, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    total_sentences = sum(len(paper['sentences']) for paper in predictions_data)
    correct_predictions = sum(
        1 for paper in predictions_data
        for sent in paper['sentences']
        if sent['predicted_label'] == sent['true_label']
    )
    total_positive = sum(
        1 for paper in predictions_data
        for sent in paper['sentences']
        if sent['true_label'] == 1
    )

    output_data = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'num_papers': len(predictions_data),
            'total_sentences': total_sentences,
            'statistics': {
                'accuracy': correct_predictions / total_sentences if total_sentences > 0 else 0,
                'total_correct': correct_predictions,
                'total_positive_samples': total_positive,
                'positive_ratio': total_positive / total_sentences if total_sentences > 0 else 0
            }
        },
        'predictions': predictions_data
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Total papers processed: {len(predictions_data)}")
    print(f"Total sentences: {total_sentences}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {output_data['metadata']['statistics']['accuracy']:.4f}")

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=30):
    model_save_path = f"/home/kzlab/muse/Savvy/Data_collection/output/fuxian/no_window/{date_str}"
    predictions_save_path = os.path.join(model_save_path, "predictions")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(predictions_save_path, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_val_accuracy = 0

    history = {
        'train_acc': [], 'train_prec': [], 'train_recall': [], 'train_f1': [],
        'val_acc': [], 'val_prec': [], 'val_recall': [], 'val_f1': []
    }

    gradient_accumulation_steps = 4

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_train_preds.append(preds.view(-1))
            all_train_labels.append(labels.view(-1))
            train_loss += loss.item()

        if all_train_preds:
            all_train_preds_tensor = torch.cat(all_train_preds)
            all_train_labels_tensor = torch.cat(all_train_labels)
        else:
            all_train_preds_tensor = torch.tensor([], device=device)
            all_train_labels_tensor = torch.tensor([], device=device)

        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
            all_train_preds_tensor, all_train_labels_tensor
        )
        avg_train_loss = train_loss / max(1, len(train_dataloader))

        # 验证
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []

        predictions_data = collect_predictions(model, val_dataloader, device)
        prediction_file = os.path.join(predictions_save_path, f'predictions_epoch_{epoch+1}.json')
        save_predictions_to_json(predictions_data, prediction_file)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_val_preds.append(preds.view(-1))
                all_val_labels.append(labels.view(-1))

        if all_val_preds:
            all_val_preds_tensor = torch.cat(all_val_preds)
            all_val_labels_tensor = torch.cat(all_val_labels)
        else:
            all_val_preds_tensor = torch.tensor([], device=device)
            all_val_labels_tensor = torch.tensor([], device=device)

        val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(
            all_val_preds_tensor, all_val_labels_tensor
        )
        avg_val_loss = val_loss / max(1, len(val_dataloader))

        print('\nTraining Metrics:')
        print(f'Loss: {avg_train_loss:.4f}')
        print(f'Accuracy: {train_accuracy:.4f}')
        print(f'Precision: {train_precision:.4f}')
        print(f'Recall: {train_recall:.4f}')
        print(f'F1 Score: {train_f1:.4f}')

        print('\nValidation Metrics:')
        print(f'Loss: {avg_val_loss:.4f}')
        print(f'Accuracy: {val_accuracy:.4f}')
        print(f'Precision: {val_precision:.4f}')
        print(f'Recall: {val_recall:.4f}')
        print(f'F1 Score: {val_f1:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print(f'\nNew best model saved with validation accuracy: {val_accuracy:.4f}')

        print('\n' + '='*70)

        history['train_acc'].append(train_accuracy)
        history['train_prec'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_acc'].append(val_accuracy)
        history['val_prec'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        if (epoch + 1) % 5 == 0:
            plot_metrics(history, epoch + 1)

    return history

def main():
    # 路径设置
    train_file_path = "/home/kzlab/muse/Savvy/Data_collection/dataset/dataset-descrption/dataset/json/1212.json"
    bert_based_cased_path = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("Starting to load data...")
    papers = load_data(train_file_path)
    print(f"Loaded {len(papers)} papers")

    print("Splitting data into train and validation sets...")
    train_papers, val_papers = train_test_split(papers, test_size=0.2, random_state=42)
    print(f"Train set size: {len(train_papers)}, Validation set size: {len(val_papers)}")

    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(bert_based_cased_path)
    config = BertConfig.from_pretrained(bert_based_cased_path, num_labels=2)

    # 直接使用官方二分类模型（逐句）
    model = BertForSequenceClassification.from_pretrained(
        bert_based_cased_path,
        config=config
    ).to(device)
    print("Model initialized successfully")

    print("Creating datasets (sentence-level)...")
    train_dataset = PaperDataset(papers=train_papers, tokenizer=tokenizer, max_length=512)
    val_dataset = PaperDataset(papers=val_papers, tokenizer=tokenizer, max_length=512)
    print("Datasets created successfully")

    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    print("Dataloaders created successfully")

    print("Starting training...")
    train_model(model, train_dataloader, val_dataloader, device)

if __name__ == '__main__':
    main()
