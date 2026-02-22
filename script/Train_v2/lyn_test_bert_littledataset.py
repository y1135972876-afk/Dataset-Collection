"""
该代码用于计算模型在测试集上的指标
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
import json
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append("/home/kzlab/muse/Savvy/Data_collection/dataset/dataset-descrption")  # 添加项目根目录到 Python 路径

# 导入你的自定义类（确保这些文件可以被导入）
from correct_bert import (
    PaperDataset, 
    BertForSentenceClassification, 
    custom_collate_fn,
    calculate_metrics,
    load_data,
    save_predictions_to_json
)

def evaluate_model(model, test_dataloader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0
    predictions_data = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            paper_names = batch['paper_names']
            sentences = batch['sentences']
            indices = batch['indices']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss, logits = outputs
            test_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, max_sentences]
            probs = torch.softmax(logits, dim=-1)

            # 收集预测数据
            for b in range(len(paper_names)):
                valid_indices = indices[b][indices[b] != -1]
                valid_predictions = predictions[b][:len(valid_indices)]
                valid_probs = probs[b][:len(valid_indices)]
                valid_labels = labels[b][:len(valid_indices)]
                valid_sentences = sentences[b][:len(valid_indices)]

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

    # 计算平均损失
    avg_loss = test_loss / len(test_dataloader)

    # 计算评估指标
    accuracy, precision, recall, f1 = calculate_metrics(
        torch.tensor(all_predictions),
        torch.tensor(all_labels)
    )

    # 生成详细的分类报告
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=['Class 0', 'Class 1'],
        digits=4
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

def main():
    # 设置路径
    date_str = "1225"
    test_file_path = "/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json"  # 测试集文件路径
    model_path = "/home/kzlab/muse/Savvy/Data_collection/output/fuxian/1224/best_model.pth"  # 已训练模型路径
    bert_base_cased_path = "/home/kzlab/muse/Savvy/Data_collection/models/bert-base-cased"  # BERT预训练模型路径
    output_dir = f"/home/kzlab/muse/Savvy/Data_collection/output/revised/revised_epoch"  # 测试结果输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载测试数据
    print("Loading test data...")
    test_papers = load_data(test_file_path)
    print(f"Loaded {len(test_papers)} test papers")

    # 初始化tokenizer和模型
    print("Initializing model...")
    tokenizer = BertTokenizer.from_pretrained(bert_base_cased_path)
    config = BertConfig.from_pretrained(bert_base_cased_path, num_labels=2)
    
    # 创建模型实例
    model = BertForSentenceClassification(config, tokenizer=tokenizer)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded successfully")

    # 创建测试数据集和数据加载器
    print("Creating test dataloader...")
    test_dataset = PaperDataset(
        papers=test_papers,
        tokenizer=tokenizer,
        max_length=512,
        min_sentences=3,
        context_size=2
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # 评估模型
    print("Starting evaluation...")
    results = evaluate_model(model, test_dataloader, device)

    # 打印评估结果
    print("\nTest Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(results['report'])

    # 保存预测结果
    predictions_file = os.path.join(output_dir, 'test_predictions.json')
    save_predictions_to_json(results['predictions'], predictions_file)

    # 保存评估指标
    metrics_file = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'loss': results['loss'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'classification_report': results['report']
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main()