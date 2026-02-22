import requests
from bs4 import BeautifulSoup
import json
import re
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
import logging
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import GrobidParser
from openai import OpenAI
from itertools import groupby
from typing import List, Dict
import queue
from tqdm import tqdm   # ✅ 新增进度条库
from Project1.utils import *
from Project1.prompt import *


# 基础URL
base_url = "https://arxiv.org"

# 要爬取的类别
categories = [
    'cs.IR',
    'cs.DB',
    'cs.AI',
    'cs.CL',
    'cs.CV',
    'cs.MA'
]


def get_date(year=None, month=None, day=None):
    """
    获取指定日期并格式化
    参数:
        year: 年份，默认当前年份
        month: 月份，默认当前月份
        day: 日期，默认当前日期
    返回:
        格式化的日期字符串 (YYYY-MM-DD)
    """
    if all(x is None for x in (year, month, day)):
        return datetime.now().strftime('%Y-%m-%d')
    
    current = datetime.now()
    year = year if year is not None else current.year
    month = month if month is not None else current.month
    day = day if day is not None else current.day
    return datetime(year, month, day).strftime('%Y-%m-%d')


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 固定随机性
def set_all_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    

# 处理数据函数 ： 添加content属性
def process_data_fun(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if 'url' in item:
            del item['url'] #为什么删除url，影响bert判断了吗？
        item['content'] = item["title"] + ' ' + item["abstract"]
        del item['title']
        del item['abstract']

    processed_data = []
    for item in data:
        if 'label' not in item:
            if item['content'].strip():
                item['content'] = item['content'].strip()
                processed_data.append(item)
        else:
            try:
                if item['content'].strip():
                    item['label'] = int(item['label'])
                    item['content'] = item['content'].strip()
                    processed_data.append(item)
            except ValueError:
                print(f"Invalid label: {item['label']} - Skipping this entry")

    return processed_data


class PDFDataset(Dataset):
    def __init__(self, data, tokenizer, shuffle=False):
        self.data = data
        self.tokenizer = tokenizer
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        #传入的text只是content
        text = item["content"]
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, padding="max_length",
            max_length=512, add_special_tokens=True
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        return item

    def shuffle_data(self):
        if self.shuffle:
            import random
            random.shuffle(self.data)


def test_model(test_dataset, tokenizer, model_save_path, epoch):
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    #batchsize是2,2 个一输入，预测
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    all_results = []

    # ✅ 使用 tqdm 展示 batch 进度
    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Inference", unit="batch")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)


        '''
        每次迭代中，批次的样本是一起推理的，probs 是一个 2 维张量，其中包含了每个样本的概率。
        enumerate(probs.cpu().numpy()) 是对每个批次内的每个样本逐一处理的，所以对于每个批次里的 2 个样本，它会分开进行处理：
        i 会依次是 0 和 1（即每批次的 2 个样本）。
        由于 batch_idx 和 batch_size 的组合，每个样本的 Sample 编号是全局唯一的，batch_idx * batch_size + i 确保了无论有多少个批次，所有样本都有独立的编号。
        '''
        for i, prob in enumerate(probs.cpu().numpy()):
            text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            #构建result的时候都构建，保存的时候只保存概率大于0.5的
            result = {
                'Sample': batch_idx * test_dataloader.batch_size + i,
                'Text': text[:80] + "..." if len(text) > 80 else text,
                'Predicted_Probabilities': prob.tolist()
            }
            if result['Predicted_Probabilities'][1] >= 0.5:
                all_results.append(result)

        # ✅ 每 10 个 batch 打印一次调试信息
        if batch_idx % 10 == 0:
            print(f"[DEBUG] Batch {batch_idx}, logits shape={logits.shape}, sample_probs={probs[0].tolist()}")

    return all_results


def pred_label():
    tokenizer = BertTokenizer.from_pretrained('/home/kemove/project/DC/models/fine_tuned_model')
    set_all_seeds(42)
    today = get_date(2025, 9, 17)
    model_save_path = '/home/kemove/project/DC/models/checkpoint-1530'

    # ✅ tqdm 展示类别进度
    for category in tqdm(categories, desc="Categories", unit="category"):
        data = process_data_fun(f"/home/kemove/project/DC/output/1_paper/{today}/{category}.json")
        dataset = PDFDataset(data, tokenizer, shuffle=False)
        results = test_model(dataset, tokenizer, model_save_path, "fine_tune")

        # 确保输出目录存在
        output_dir = f'/home/kemove/project/DC/output/2_paper_processed/{today}'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f'process_{category}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # ✅ 增加统计信息
        print(f"[INFO] {category}: total={len(dataset)}, detected={len(results)}, saved to {output_file}")


if __name__ == "__main__":
    pred_label()
