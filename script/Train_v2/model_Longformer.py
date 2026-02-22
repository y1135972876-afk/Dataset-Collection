"""
更正过的bert训练文！！！！！！！！！！！！！！！！
用这个进行训练！
不需要管段落之间的联系
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只让物理卡1可见
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb=128"

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerModel, LongformerConfig
# from transformers import AdamW
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
from datetime import datetime
from torch import nn

date_str = "1224"

# ——————————————————————计算指标和绘图——————————————————————
def calculate_metrics(predictions, labels):
    """计算评估指标 - 正确处理GPU/CPU转换"""
    # 忽略填充的标签（-100）
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # 关键：将GPU tensor临时转移到CPU进行计算
    # 但这不影响训练过程仍在GPU上进行
    if predictions.is_cuda:
        # 如果predictions在GPU上，转移到CPU
        predictions_cpu = predictions.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
    else:
        # 如果已经在CPU上，直接转换
        predictions_cpu = predictions.numpy()
        labels_cpu = labels.numpy()
    
    # 处理空数组的情况
    if len(predictions_cpu) == 0 or len(labels_cpu) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 在CPU上计算scikit-learn指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_cpu,
        predictions_cpu,
        average='binary',
        zero_division=0
    )
    
    # 准确率可以直接用PyTorch计算（支持GPU）
    accuracy = (predictions == labels).float().mean().item()
    
    return accuracy, precision, recall, f1

def plot_metrics(history, epoch):
    fig_save_path = f"/home/kzlab/muse/Savvy/Data_collection/output/revised/revised_model/longformer/{date_str}"
    
    os.makedirs(fig_save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 设置子图
    metrics = ['acc', 'prec', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 2, i)
        
        # 绘制训练集曲线
        plt.plot(history[f'train_{metric}'], label='Train', marker='o')
        # 绘制验证集曲线
        plt.plot(history[f'val_{metric}'], label='Validation', marker='o')
        
        plt.title(f'{title} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, f'metrics_epoch_{epoch}.png'))
    plt.close()


# ——————————————————————计算指标和绘图——————————————————————
class PaperDataset(Dataset):
    def __init__(self, papers, tokenizer, max_length=4096, min_sentences=7, context_size=6):
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
            # 计算当前句子的tokens
            tokens = self.tokenizer.encode(sentences[i]['text'], add_special_tokens=False)
            next_length = current_tokens + len(tokens) + (1 if current_tokens > 0 else 2)
            
            # 检查是否超过最大长度
            if next_length >= self.max_length - 2:
                break
            
            current_tokens = next_length
            window_size += 1
        
        return max(self.min_sentences, window_size) # min_sentences限制了窗口的最小句子数
    
    def _prepare_examples(self):
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
    
    def __len__(self):
        return len(self.flat_examples)
    
    def __getitem__(self, idx):
        example = self.flat_examples[idx]
        sentences = example['sentences']
        
        # 获取句子文本和标签
        text_list = [sent['text'] for sent in sentences]
        labels = [sent['label'] for sent in sentences]
        
        # 使用Longformer的特殊标记连接句子
        full_text = ' </s> '.join(text_list) # 使用Longformer的句子分隔符
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
            'indices': torch.tensor(example['indices'], dtype=torch.long),
            'paper_name': example['paper_name'],
            'has_positive': example['has_positive'],
            'sentences': text_list  # 添加这一行
        }

def create_dataloader(papers, tokenizer, batch_size=4, max_length=4096):
    """创建数据加载器的辅助函数"""
    dataset = PaperDataset(
        papers=papers,
        tokenizer=tokenizer,
        max_length=max_length,
        min_sentences=7,
        context_size=6
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )       
        
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
            sentences = batch['sentences']  # 这里包含完整的句子文本
            
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
                for idx, pred, prob, label, sent_text in zip(valid_indices, 
                                                           valid_predictions, 
                                                           valid_probs,
                                                           valid_labels,
                                                           valid_sentences):
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
      
        
def custom_collate_fn(batch):
    input_ids = []
    attention_masks = []
    labels = []
    indices = []
    paper_names = []
    sentences_list = []  # 新增
    
    max_sentences = max(len(item['labels']) for item in batch) # 看看这个batch里，输入的最多的句子数有多少
    
    for item in batch:
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
        
        # 填充标签到相同长度
        curr_labels = item['labels']
        if len(curr_labels) < max_sentences: #防止出现标签长度和句子数不同的情况
            padding = torch.full((max_sentences - len(curr_labels),), -100, dtype=curr_labels.dtype)
            curr_labels = torch.cat([curr_labels, padding])
        labels.append(curr_labels)
        
        curr_indices = item['indices']
        if len(curr_indices) < max_sentences:
            padding = torch.full((max_sentences - len(curr_indices),), -1, dtype=curr_indices.dtype)
            curr_indices = torch.cat([curr_indices, padding])
        indices.append(curr_indices)
        
        paper_names.append(item['paper_name'])
        sentences_list.append(item['sentences'])  # 新增
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels),
        'indices': torch.stack(indices),
        'paper_names': paper_names,
        'sentences': sentences_list  # 新增
    }

class LongformerForSentenceClassification(nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.longformer = LongformerModel(config)
        self.config = config
        
        # 句子分类器
        self.sentence_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # 二分类
        )
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化分类器权重"""
        for module in self.sentence_classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]
        
        batch_size = input_ids.size(0)
        max_sentences = labels.size(1) if labels is not None else 1
        
        logits_list = []
        
        for i in range(batch_size):
            # 使用Longformer的句子分隔符（</s>）来分割句子
            sep_positions = (input_ids[i] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) == 0:
                # 如果没有找到SEP标记，使用整个序列
                sentence_representations = [sequence_output[i, 0]]  # 使用<s>标记
            else:
                # 提取每个句子的表示
                sentence_representations = []
                
                # 第一个句子：<s>到第一个</s>
                start_pos = 0
                end_pos = sep_positions[0] if len(sep_positions) > 0 else sequence_output.size(1)
                first_sentence = sequence_output[i, start_pos:end_pos+1]
                # 对第一个句子进行平均池化
                if len(first_sentence) > 0:
                    first_sentence_rep = first_sentence.mean(dim=0)
                    sentence_representations.append(first_sentence_rep)
                
                # 后续句子
                for j in range(len(sep_positions)):
                    if j < len(sep_positions) - 1:
                        start_pos = sep_positions[j] + 1
                        end_pos = sep_positions[j + 1]
                    else:
                        # 最后一个句子
                        start_pos = sep_positions[j] + 1
                        end_pos = sequence_output.size(1) - 1
                    
                    if start_pos < end_pos:
                        sentence_tokens = sequence_output[i, start_pos:end_pos]
                        if len(sentence_tokens) > 0:
                            sentence_rep = sentence_tokens.mean(dim=0)
                            sentence_representations.append(sentence_rep)
            
            # 处理句子表示
            if sentence_representations:
                sentence_tensor = torch.stack(sentence_representations)
                current_sentences = len(sentence_representations)
                
                # 填充或截断到最大句子数
                if current_sentences < max_sentences:
                    padding_size = max_sentences - current_sentences
                    padding = torch.zeros(padding_size, sentence_tensor.size(-1)).to(input_ids.device)
                    sentence_tensor = torch.cat([sentence_tensor, padding], dim=0)
                else:
                    sentence_tensor = sentence_tensor[:max_sentences]
                
                # 对每个句子进行分类
                sentence_logits = self.sentence_classifier(sentence_tensor)
                logits_list.append(sentence_logits)
            else:
                # 如果没有句子，创建空的logits
                empty_logits = torch.zeros(max_sentences, 2).to(input_ids.device)
                logits_list.append(empty_logits)
        
        # 堆叠所有预测
        padded_logits = torch.stack(logits_list)  # [batch_size, num_sentences, 2]
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            active_logits = padded_logits.view(-1, 2)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
            
        return loss, padded_logits
        

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

def save_predictions_to_json(predictions_data, output_path):
    """将预测结果保存为JSON文件，包含更详细的统计信息"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算统计信息
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
    
    # 添加元数据和统计信息
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

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=10):
    # 路径设置
    model_save_path = f"/home/kzlab/muse/Savvy/Data_collection/output/revised/revised_model/longformer/{date_str}"
    predictions_save_path = os.path.join(model_save_path, "predictions")
    os.makedirs(model_save_path, exist_ok=True)  # 确保目录存在
    os.makedirs(predictions_save_path, exist_ok=True)  # 确保预测目录也存在
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    best_val_accuracy = 0
    
    # 存储每个epoch的指标
    history = {
        'train_acc': [], 'train_prec': [], 'train_recall': [], 'train_f1': [],
        'val_acc': [], 'val_prec': [], 'val_recall': [], 'val_f1': []
    }
    
    gradient_accumulation_steps = 4  # 累积4个批次的梯度再更新
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_preds = []  # 改为收集tensor列表
        all_train_labels = []  # 改为收集tensor列表
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs[0] / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 获取每个句子的预测结果
            predictions = torch.argmax(outputs[1], dim=-1)  # shape: [batch_size, max_sentences]
            
            # 修改：直接收集GPU tensor，不转换为numpy
            all_train_preds.append(predictions.view(-1))
            all_train_labels.append(labels.view(-1))
            
            train_loss += loss.item()
        
        # 堆叠所有tensor
        if all_train_preds:
            all_train_preds_tensor = torch.cat(all_train_preds)
            all_train_labels_tensor = torch.cat(all_train_labels)
        else:
            all_train_preds_tensor = torch.tensor([], device=device)
            all_train_labels_tensor = torch.tensor([], device=device)
        
        # 计算训练集指标
        train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
            all_train_preds_tensor,
            all_train_labels_tensor
        )
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_val_preds = []  # 改为收集tensor列表
        all_val_labels = []  # 改为收集tensor列表
        
        predictions_data = collect_predictions(model, val_dataloader, device)
        
        # 保存预测结果
        prediction_file = os.path.join(
            predictions_save_path, 
            f'predictions_epoch_{epoch+1}.json'
        )
        save_predictions_to_json(predictions_data, prediction_file)        
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs[0].item()
                
                # 获取每个句子的预测结果
                predictions = torch.argmax(outputs[1], dim=-1)  # shape: [batch_size, max_sentences]
                
                # 修改：直接收集GPU tensor，不转换为numpy
                all_val_preds.append(predictions.view(-1))
                all_val_labels.append(labels.view(-1))
        
        # 堆叠验证集的tensor
        if all_val_preds:
            all_val_preds_tensor = torch.cat(all_val_preds)
            all_val_labels_tensor = torch.cat(all_val_labels)
        else:
            all_val_preds_tensor = torch.tensor([], device=device)
            all_val_labels_tensor = torch.tensor([], device=device)
        
        # 计算验证集指标
        val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(
            all_val_preds_tensor,
            all_val_labels_tensor
        )
        avg_val_loss = val_loss / len(val_dataloader)
        
        # 打印详细的训练信息
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
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print(f'\nNew best model saved with validation accuracy: {val_accuracy:.4f}')
        
        print('\n' + '='*70)
        
        # 存储指标
        history['train_acc'].append(train_accuracy)
        history['train_prec'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_acc'].append(val_accuracy)
        history['val_prec'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # 每5个epoch画一次图
        if (epoch + 1) % 1 == 0:
            plot_metrics(history, epoch + 1)
            
    return history  # 返回训练历史，方便后续分析



def main():
    # 1 所有路径设置
    train_file_path = "/home/kzlab/muse/Savvy/Data_collection/dataset/dataset-descrption/dataset/json/1212.json"
    longformer_path = "/home/kzlab/muse/Savvy/Data_collection/models/models--allenai--longformer-base-4096/snapshots/301e6a42cb0d9976a6d6a26a079fef81c18aa895"
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 2 加载数据
    print("Starting to load data...") 
    papers = load_data(train_file_path)
    print(f"Loaded {len(papers)} papers")
    # 划分训练集和验证集
    print("Splitting data into train and validation sets...")
    train_papers, val_papers = train_test_split(papers, test_size=0.2, random_state=42) # 有name，sentences等属性
    print(f"Train set size: {len(train_papers)}, Validation set size: {len(val_papers)}")
    
    # 3 初始化tokenizer和模型
    print("Initializing tokenizer and model...")
    tokenizer = LongformerTokenizer.from_pretrained(longformer_path)
    # 初始化配置
    config = LongformerConfig.from_pretrained(longformer_path)
    config.num_labels = 2
    
    # 创建模型实例，传入tokenizer
    model = LongformerForSentenceClassification(config, tokenizer=tokenizer)
    model = model.to(device)
    print("Model initialized successfully")
    
    # 4 创建数据集
    print("Creating datasets...")
    train_dataset = PaperDataset( 
        papers=train_papers,
        tokenizer=tokenizer,
        max_length=4096,
        min_sentences=7,
        context_size=6  # 先创建Dataset：为每个正样本保留的上下文句子数
    )
    val_dataset = PaperDataset(
        papers=val_papers,
        tokenizer=tokenizer,
        max_length=4096,
        min_sentences=7,
        context_size=6
    )
    print("Datasets created successfully")
    
    # 创建数据加载器
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    ) # 再创建DataLoader
    print("Dataloaders created successfully")
    
    # 5 训练模型
    print("Starting training...")
    train_model(model, train_dataloader, val_dataloader, device)

if __name__ == '__main__':
    main()