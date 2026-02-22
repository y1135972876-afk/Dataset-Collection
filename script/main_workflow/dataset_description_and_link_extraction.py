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
from Project1.utils import *
from Project1.prompt import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # 如果没有提供任何参数，返回今天的日期
        return datetime.now().strftime('%Y-%m-%d')
    
    # 获取当前日期作为默认值
    current = datetime.now()
    year = year if year is not None else current.year
    month = month if month is not None else current.month
    day = day if day is not None else current.day
    
    return datetime(year, month, day).strftime('%Y-%m-%d')

def extract_json_values(file_path, key):
    """
    从JSON文件中提取指定key的值，并去重
    file_path: JSON文件路径
    key: 要提取的键名
    """
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 未找到。")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            values = []
            seen = set()  # 用于去重的集合

            # 处理数据字典
            def process_dict(item):
                if isinstance(item, dict):
                    # 如果是data字段，特殊处理
                    if 'data' in item:
                        for data_item in item['data']:
                            if isinstance(data_item, dict):
                                # 获取sentences字段
                                sentences = data_item.get('sentences', [])
                                for sentence in sentences:
                                    if isinstance(sentence, dict):
                                        text = sentence.get('text', '')
                                        if text and text not in seen:
                                            seen.add(text)
                                            values.append(text)
                    # 获取指定key的值
                    value = item.get(key)
                    if value and value not in seen:
                        seen.add(value)
                        values.append(value)

            # 根据数据类型处理
            if isinstance(data, list):
                for item in data:
                    process_dict(item)
            elif isinstance(data, dict):
                process_dict(data)

            return values  # 返回去重后的值列表

    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是一个有效的JSON文件。")
        return []
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误：{str(e)}")
        return []

def find_json_url(file_path, values_queue, category):
    if not values_queue:
        print("该类没有自建数据集论文！")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pdf_url = []
        for value in values_queue:
            for item in data:
                if item.get('count') == value:
                    pdf_url.append(item['url'])
        return pdf_url
        
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误：{str(e)}")
        return []  


def download_pdf(url, save_path):
    """下载PDF文件到指定路径。"""
    # 转换URL格式为PDF下载链接
    if 'arxiv.org/html' in url:
        url = url.replace('/html/', '/pdf/') + '.pdf'
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"成功下载 {url} 到 {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"下载 {url} 时发生错误：{e}")
    except Exception as e:
        print(f"保存文件 {save_path} 时发生错误：{e}")

def test_grobid_connection(endpoint):
    """测试GROBID服务是否正常运行"""
    try:
        response = requests.get(endpoint.replace('/processFulltextDocument', '/isalive'))
        if response.status_code == 200:
            logger.info("GROBID服务正常运行")
            return True
        else:
            logger.error(f"GROBID服务返回状态码: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"无法连接到GROBID服务: {str(e)}")
        return False


def test_pdf_processing(pdf_path: str, file_name: str) -> bool:
    """用 GROBID 提取 PDF 文本并写入同一个 TXT 文件；不同 doc 之间仅换行分隔"""
    GROBID_ENDPOINT = "http://localhost:8071/api/processFulltextDocument"

    # 1) 先检查 GROBID 服务
    if not test_grobid_connection(GROBID_ENDPOINT):
        return False

    try:
        logger.info(f"处理文件: {os.path.join(pdf_path, file_name)}")

        # 用 splitext 正确去掉扩展名（而不是 strip）
        base_name, ext = os.path.splitext(file_name)
        if ext.lower() != ".pdf":
            logger.warning(f"传入的文件名不是 .pdf：{file_name}")

        # 2) 解析 PDF
        loader = GenericLoader.from_filesystem(
            pdf_path,
            glob=file_name,            # 精确匹配该 PDF
            suffixes=[".pdf"],
            parser=GrobidParser(
                segment_sentences=False,
                grobid_server=GROBID_ENDPOINT
            )
        )
        logger.info("开始加载PDF文档...")
        docs = loader.load()

        if not docs:
            logger.warning("没有从PDF中提取到任何内容")
            return False

        logger.info(f"成功提取了 {len(docs)} 个文档")

        # 3) 取一个论文标题（优先 metadata.paper_title）
        paper_title = next(
            ( (d.metadata or {}).get("paper_title") for d in docs
              if (d.metadata or {}).get("paper_title") ),
            base_name
        )
        safe_title = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in paper_title).strip()

        # 4) 收集文本：空的跳过；不同 doc 之间只加一个换行
        chunks = []
        for i, doc in enumerate(docs, 1):
            content = (doc.page_content or "").strip()
            if not content:
                logger.warning(f"文档 {i} 提取内容为空，跳过")
                continue
            chunks.append(content)

        if not chunks:
            logger.warning("所有文档内容都为空，放弃写入")
            return False

        combined_text = f"[TITLE]{safe_title}[/TITLE]\n" + "\n".join(chunks) + "\n"

        # 5) 先清空文件再写入新内容
        output_file_path = os.path.join(pdf_path, f"{base_name}.txt")
        
        # 先清空文件内容
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("")  # 写入空字符串以清空文件
        
        # 然后写入新内容
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(combined_text)

        logger.info(f"已更新文本至: {output_file_path}")
        preview = combined_text[:100].replace("\n", " ").strip()
        logger.info(f"内容预览: {preview}...")
        return True

    except Exception as e:
        logger.error(f"处理PDF时发生错误: {str(e)}")
        return False



def create_new_json(file_path, file_name):
    """
    在指定路径创建新的JSON文件
    """
    # 确保目录存在
    os.makedirs(file_path, exist_ok=True)
    
    # 完整的文件路径
    full_path = os.path.join(file_path, f"{file_name}")
    
    # 创建初始JSON结构
    initial_data = {
        "data": []
    }
    
    # 写入文件
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=4)
        print(f"成功创建JSON文件: {full_path}")
        return full_path
    except Exception as e:
        print(f"创建JSON文件时出错: {str(e)}")
        return None


# ——————————————————————————————同属于process_single_file——————————————————————————————————————————
def merge_lines_in_block(lines, min_length=100):
    """
    对块中的行按逻辑进行合并。
    """
    merged_lines = []
    buffer = ""

    # 正则表达式用于检测特定规则
    merge_condition_colon = re.compile(r":\s*$")
    lowercase_start = re.compile(r"^[a-z]")
    numbered_or_bullet_start = re.compile(r"^(\s*\d+\.|\•|-|\*)")
    punctuation_start = re.compile(r'^[,.;，?]')  # 检测以标点符号开头的行
    digit_start = re.compile(r'^\d')  # 检测以数字开头的行

    i = 0
    while i < len(lines):
        stripped_line = lines[i].strip()
        if (punctuation_start.match(stripped_line) or lowercase_start.match(stripped_line)) and merged_lines:
            merged_lines[-1] += " " + stripped_line
            i += 1
        elif merge_condition_colon.search(stripped_line):
            buffer += " " + stripped_line if buffer else stripped_line
            i += 1
            if i < len(lines) and (numbered_or_bullet_start.match(lines[i].strip()) or len(lines[i].strip()) < min_length):
                while i < len(lines) and (numbered_or_bullet_start.match(lines[i].strip()) or len(lines[i].strip()) < min_length):
                    buffer += " " + lines[i].strip()
                    i += 1
                merged_lines.append(buffer)
                buffer = ""
            else:
                if i < len(lines):
                    next_line = lines[i].strip()
                    buffer += " " + next_line if buffer else next_line
                    merged_lines.append(buffer)
                    buffer = ""
                    i += 1
                else:
                    merged_lines.append(buffer)
                    buffer = ""
        elif i + 1 < len(lines) and lowercase_start.match(lines[i + 1].strip()):
            buffer = stripped_line
            buffer += " " + lines[i + 1].strip()
            i += 2
            merged_lines.append(buffer)
            buffer = ""
        elif len(stripped_line) < min_length:
            if digit_start.match(stripped_line) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                merged_lines.append(stripped_line + " " + next_line)
                i += 2
            else:
                if merged_lines:
                    merged_lines[-1] += " " + stripped_line
                    i += 1
                else:
                    merged_lines.append(stripped_line)
                    i += 1
        else:
            merged_lines.append(stripped_line)
            i += 1

    if buffer:
        merged_lines.append(buffer)

    return merged_lines




def merge_lines_in_file(input_file_path, output_file_path, min_length=100):
    """
    按标题分块处理文件中的行。
    """
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    merged_lines = []
    current_block = []

    # 标题检测正则
    title_pattern = re.compile(r"\[TITLE\].*\[/TITLE\]")
    #通过规则判断是否需要将某些行合并到一起，以避免拆散和断开不完整的句子或者段落。
    for line in lines:
        stripped_line = line.strip()

        # 如果是标题行
        if title_pattern.match(stripped_line):
            # 处理当前块中的内容
            if current_block:
                merged_lines.extend(merge_lines_in_block(current_block, min_length))
                current_block = []
            # 添加标题
            merged_lines.append(stripped_line)
        else:
            # 非标题行，加入当前块
            current_block.append(stripped_line)

    # 处理最后一个块
    if current_block:
        merged_lines.extend(merge_lines_in_block(current_block, min_length))

    # 写入输出文件
    with open(output_file_path, "w", encoding="utf-8") as file:
        for merged in merged_lines:
            file.write(merged + "\n")

def add_title_to_sentences_with_full_title(input_file_path, output_file_path):
    """
    安全版本：为每个句子添加对应的完整标题，不修改原文件。
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    title_pattern = re.compile(r"\[TITLE\](.*?)\[/TITLE\]")
    current_title = None
    updated_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        match = title_pattern.match(stripped_line)
        if match:
            current_title = stripped_line
        elif current_title and stripped_line:
            updated_lines.append(f"{current_title}{stripped_line}")
        elif stripped_line:
            # 处理没有标题的内容
            updated_lines.append(f"[TITLE]Unknown[/TITLE]{stripped_line}")
    
    # 如果没有处理任何内容，复制原始内容
    if not updated_lines:
        updated_lines = [line.strip() for line in lines if line.strip()]
    # 写入输出文件（不覆盖原文件）
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for updated_line in updated_lines:
            outfile.write(updated_line + "\n")
    
    return len(updated_lines) > 0

def process_single_file(input_folder, output_folder, target_filename):
    """
    处理文件夹中的指定TXT文件。
    
    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
        target_filename (str): 要处理的目标文件名
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建完整的文件路径
    input_file_path = os.path.join(input_folder, target_filename)
    output_file_path = os.path.join(output_folder, target_filename)

    # 检查文件是否存在
    if not os.path.exists(input_file_path):
        print(f"错误：文件 {target_filename} 不存在于 {input_folder} 目录中")
        return False

    # 检查文件是否是txt文件
    if not target_filename.endswith('.txt'):
        print(f"错误：文件 {target_filename} 不是txt文件")
        return False

    try:
        print(f"正在处理文件: {target_filename}")
        #主要处理合并文章的路径
        merge_lines_in_file(input_file_path, output_file_path)
        add_title_to_sentences_with_full_title(output_file_path,output_file_path)
        print("处理完成！")
        return True
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return False            

# ——————————————————————————————END：同属于process_single_file——————————————————————————————————————————



# ——————————————————————————————同属于process_txt_to_json——————————————————————————————————————————
def get_title_from_text(text):
    """
    从文本中提取第一个标题作为论文名称
    """
    pattern = r'\[TITLE\](.*?)\[/TITLE\]'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

def extract_text_without_title(text):
    """
    移除文本中的[TITLE]...[/TITLE]部分，返回纯文本内容
    """
    # 使用正则表达式移除所有标题部分
    pattern = r'\[TITLE\].*?\[/TITLE\]'
    return re.sub(pattern, '', text).strip()

def split_sentences(text):
    """改进的句子分割函数"""
    # 处理常见缩写
    abbreviations = [
        'e.g.', 'i.e.', 'etc.', 'vs.', 'fig.', 'Fig.',
        'al.', 'cf.', 'v.', 'vs.', 'Mr.', 'Mrs.', 'Ms.',
        'Dr.', 'Prof.', 'Sr.', 'Jr.', 'Co.', 'Ltd.',
        'St.', 'Ph.D.', 'U.S.', 'U.K.', 'U.N.'
    ]
    
    # 临时替换缩写中的点号
    temp_text = text
    for abbr in abbreviations:
        temp_text = temp_text.replace(abbr, abbr.replace('.', '@'))
    
    # 在句号后面添加空格（如果没有的话）
    temp_text = re.sub(r'\.([A-Z])', r'. \1', temp_text)
    
    # 使用正则表达式分割句子
    # 匹配模式：句子以句号、感叹号或问号结束，后面可能跟着任意空白字符，然后是大写字母
    pattern = r'[^.!?]+[.!?](?=\s+|[A-Z]|$)'
    sentences = re.findall(pattern, temp_text)
    
    # 恢复缩写中的点号，清理空白字符
    sentences = [s.replace('@', '.').strip() for s in sentences]
    
    # 过滤空句子并确保每个句子都以句号结束
    final_sentences = []
    for s in sentences:
        if s.strip():
            sentence = s.strip()
            if not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
                sentence += '.'
            final_sentences.append(sentence)
    
    return final_sentences


def process_paragraph(text, paper_name="", section="", start_paragraph_id=None):
    """处理输入的段落文本，返回符合格式的字典"""
    # 1.按照指定的分隔符分割句子
    sentences = split_sentences(text)
    
    # 2.构建句子列表
    sentences_data = []
    for i, sentence in enumerate(sentences):
        sentence_dict = {
            "sentence_id": i,
            "text": sentence.strip(),
            "label": 0  # 默认标签为0
        }
        sentences_data.append(sentence_dict)
    
    # 3 构建段落字典
    paragraph_data = {
        "paper_name": paper_name,
        "section": section,
        "paragraph_id": start_paragraph_id if start_paragraph_id is not None else 0,
        "full_text": text,
        "sentences": sentences_data
    }
    
    return paragraph_data

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def process_txt_to_json(txt_file_path, output_json_path):
    """
    处理txt文件并转换为指定的JSON格式
    """
    try:
        # 读取txt文件
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 获取论文标题
        paper_name = get_title_from_text(content)
        
        # 获取不含标题的纯文本
        clean_text = extract_text_without_title(content)
        
        # 加载现有JSON或创建新的
        # json_data = load_existing_json(output_json_path)
        json_data = {"data": []}
        
        # 处理段落成一个个句子（这里我们将整个文本作为一个段落处理），默认每句话的标签为0
        paragraph_data = process_paragraph(
            text=clean_text,
            paper_name=paper_name,
            section="",  # 可以根据需要添加section
            start_paragraph_id=len(json_data['data'])  # 使用现有数据长度作为新段落的ID
        )
        
        # 添加到JSON数据中
        json_data['data'].append(paragraph_data)
        
        # 保存JSON
        save_json(json_data, output_json_path)
        
        print(f"处理完成！数据已保存到 {output_json_path}")
        
        # 打印处理后的句子供检查
        print("\n分割后的句子：")
        for i, sentence in enumerate(paragraph_data['sentences']):
            print(f"\n{i+1}. {sentence['text']}")
            
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return False

# ——————————————————————————————END：同属于process_txt_to_json——————————————————————————————————————————


class ResearchDatasetEvaluator:
    def __init__(self):
        logging.info("Initializing ResearchDatasetEvaluator")
        self.client = OpenAI(
            api_key='sk-734cdbb2ddb94134a83041a2dcf863cf',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.max_retries = 5
        self.chunk_size = 100
        
        # 设置结果文件路径
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.results_dir = f"results_{timestamp}"
        # self.papers_dir = os.path.join(self.results_dir, "papers")
        # self.final_results_file = os.path.join(self.results_dir, "final_evaluation_results.json")
        # self.interim_results_file = os.path.join(self.results_dir, "interim_results.json")
        
        # 创建必要的目录
        # os.makedirs(self.results_dir, exist_ok=True)
        # os.makedirs(self.papers_dir, exist_ok=True)     

    def load_data(self, json_str: str) -> List[Dict]:
        """加载和解析JSON数据"""
        logging.info("Starting data loading process")
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict) or 'data' not in data:
                logging.error("Invalid JSON format: missing 'data' field")
                raise ValueError("JSON数据格式错误: 缺少'data'字段")
            
            papers_data = []
            for paper_name, group in groupby(data["data"], key=lambda x: x["paper_name"]):
                logging.debug(f"Processing paper: {paper_name}")
                group_list = list(group)
                paper_data = {
                    "paper_name": paper_name,
                    "sections": [g["section"] for g in group_list],
                    "paragraphs": [g["paragraph_id"] for g in group_list],
                    "sentences": []
                }
                for g in group_list:
                    paper_data["sentences"].extend(g["sentences"])
                papers_data.append(paper_data)
                logging.debug(f"Processed {len(paper_data['sentences'])} sentences for paper {paper_name}")
            
            return papers_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            raise

    def format_paper_for_prompt(self, paper: Dict) -> str:
        """将论文内容格式化为精确的提示词"""
        logging.info(f"Formatting prompt for paper: {paper['paper_name']}")
        total_sentences = len(paper['sentences'])
        formatted_text = f"""请仔细阅读以下论文的所有内容，判断每个句子是否描述了该研究所特定构建或使用的数据集信息。

论文标题: {paper['paper_name']}

重要提示：
1. 1. 你必须为每一个句子都提供标注，不能遗漏任何句子，也不能重复标同一个句子!!!
2. 请严格按照"句子X,标记为Y。解释：Z"的格式进行标注
3. 共有{total_sentences}个句子需要标注
4. 标注完成后请检查是否所有句子都已标注,label数量与句子数量即{total_sentences}必须要能够对应的上

判断标准:
判断为1(数据集描述)的标准：
句子必须直接描述本研究特定构建的数据集,包括:
1. 该数据集的构建过程和方法
2. 该数据集的具体构成和规模
3. 该数据的来源和收集方式
4. 该数据的预处理步骤
5. 该数据集的可获取方式(如发布地址)
6. 或者其他直接或者间接描述了该数据集的句子

所有其他类型的句子均标记为0。

请按顺序判断以下每个句子，并且对每个句子给出解释，举个例子，格式为“句子103.标记为0。解释：该句描述了结论。”，即标准格式为：句子XX,标记为X。解释：XXXX:

"""
        for i, sentence in enumerate(paper['sentences'], 1):
            formatted_text += f"句子{i}: {sentence['text']}\n"

        prompt_length = len(formatted_text)
        logging.debug(f"Generated prompt length: {prompt_length} characters")
        
        # 如果提示词过长,尝试分块处理
        if prompt_length > self.chunk_size:
            logging.warning(f"Prompt exceeds chunk size ({self.chunk_size}). Consider splitting into smaller chunks.")
        
        return formatted_text
    
    def evaluate_paper(self, paper: Dict, json_file_path: str) -> None:
        """评估单篇论文并更新JSON文件中的标签"""
        logging.info(f"Processing paper: {paper['paper_name']}")
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            # 获取模型预测
            prompt = self.format_paper_for_prompt(paper)#在句子数 >100 时，前面那段 format_paper_for_prompt() 生成的大 prompt 实际上不会被用到。
            predicted_labels, token_usage = self.get_model_predictions(paper, prompt)
            
            if not predicted_labels:
                logging.warning(f"No predictions obtained for paper: {paper['paper_name']}")
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying after 30 seconds (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(30)
                    continue
                return
            
            # 检查预测标签数量是否正确
            if len(predicted_labels) != len(paper['sentences']):
                logging.warning(
                    f"Label count mismatch for {paper['paper_name']}: "
                    f"Expected {len(paper['sentences'])}, got {len(predicted_labels)}"
                )
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying after 30 seconds (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(60)
                    continue
                return
            
            # 更新JSON文件中的标签
            self.update_json_with_predictions(json_file_path, paper, predicted_labels)
            
            print(f"\n=== 处理完成: {paper['paper_name']} ===")
            print(f"总句子数: {len(paper['sentences'])}")
            print(f"预测标签数: {len(predicted_labels)}")
            print(f"Token使用情况:")
            print(f"  输入tokens: {token_usage['input_tokens']}")
            print(f"  输出tokens: {token_usage['output_tokens']}")
            print(f"  总tokens: {token_usage['total_tokens']}")
            return
        
        logging.error(f"Failed to process paper {paper['paper_name']} after {max_retries} attempts")
    
    def save_paper_result(self, paper_result: Dict, paper_idx: int) -> str:
        """保存单篇论文的评估结果到JSON文件

        Args:
            paper_result (Dict): 论文评估结果，包含指标和预测差异
            paper_idx (int): 当前论文的索引编号

        Returns:
            str: 保存的文件路径
        """
        logging.info(f"Saving results for paper {paper_result['paper_name']}")
        
        try:
            # 使用已有的save_results方法保存结果
            filepath = self.save_results(
                result=paper_result,
                result_type="paper",
                paper_idx=paper_idx,
                total_papers=paper_result['total_papers']
            )
            
            logging.info(f"Successfully saved paper results to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving paper result: {str(e)}")
            raise

    def save_results(self, result: Dict, result_type: str = "paper", paper_idx: int = None, total_papers: int = None) -> str:
        """统一的结果保存方法
        
        Args:
            result: 要保存的结果
            result_type: 结果类型 ("paper", "interim", "final")
            paper_idx: 论文索引（仅用于paper类型）
            total_papers: 总论文数（仅用于paper和interim类型）
        """
        if result_type == "paper":
            if paper_idx is None:
                raise ValueError("Paper index is required for paper results")
            
            paper_name = result["paper_name"]
            safe_name = "".join(c for c in paper_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_idx:03d}_{safe_name}.json"
            filepath = os.path.join(self.papers_dir, filename)
            
        elif result_type == "interim":
            filepath = self.interim_results_file
            result = {
                "current_paper": result,
                "overall_progress": {
                    "total_papers": total_papers,
                    "processed_papers": paper_idx,
                    "remaining_papers": total_papers - paper_idx
                }
            }
            
        else:  # final
            filepath = self.final_results_file
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"\n{result_type.capitalize()} 结果已保存至: {filepath}")
        return filepath 

    def update_json_with_predictions(self, json_file_path: str, paper_data: Dict, predicted_labels: List[int]) -> None:
        """
        更新JSON文件中的标签
        
        Args:
            json_file_path: JSON文件路径
            paper_data: 论文数据
            predicted_labels: 预测的标签列表
        """
        try:
            # 读取现有的JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 在JSON数据中找到对应的论文并更新标签
            paper_name = paper_data['paper_name']
            label_index = 0
            
            for item in json_data['data']:
                if item['paper_name'] == paper_name:
                    for sentence in item['sentences']:
                        if label_index < len(predicted_labels):
                            sentence['label'] = predicted_labels[label_index]
                            label_index += 1
            
            # 保存更新后的JSON文件
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
                
            logging.info(f"Successfully updated labels for paper: {paper_name}")
            
        except Exception as e:
            logging.error(f"Error updating JSON file: {str(e)}")

    def get_model_predictions(self, paper: Dict, prompt: str) -> tuple[List[int], Dict]:
        """使用模型获取预测标签，处理返回的文本格式，并返回token使用情况
        
        Args:
            paper: 包含论文信息的字典
            prompt: 提示词文本
            
        Returns:
            tuple: (预测标签列表, token使用统计字典)
        """
        logging.info("Starting model prediction")
        total_sentences = len(paper['sentences'])
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        # 如果句子数量小于等于100，直接处理
        if total_sentences <= 100:
            return self._process_single_chunk(prompt)
        
        # 如果大于100，进行分块处理
        chunk_size = 100
        all_labels = []
        num_chunks = (total_sentences + chunk_size - 1) // chunk_size  # 向上取整
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_sentences)
            
            # 构建当前块的提示词
            chunk_sentences = paper['sentences'][start_idx:end_idx]
            chunk_prompt = f"""请仔细阅读以下论文的所有内容，判断每个句子是否描述了该研究所特定构建或使用的数据集信息。

    论文标题: {paper['paper_name']}

    严格要求：
    [!重要!] 本次需要标注从句子{start_idx + 1}到句子{end_idx}，共{len(chunk_sentences)}个句子
    [!重要!] 你必须:
    - 按顺序标注每个句子，不能跳过
    - 每个句子只能标注一次，不能重复标注
    - 检查标注数量必须等于{len(chunk_sentences)}

    标注格式：
    句子X,标记为Y。解释：Z
    Y必须是0或1，例如"句子1,标记为0。解释：这是引言。"

    判断标准:
    判断为1(数据集描述)的标准：
    句子必须直接描述本研究特定构建的数据集,包括:
    1. 该数据集的构建过程和方法
    2. 该数据集的具体构成和规模
    3. 该数据的来源和收集方式
    4. 该数据的预处理步骤
    5. 该数据集的可获取方式(如发布地址)
    6. 或者其他直接或者间接描述了该数据集的句子

    所有其他类型的句子均标记为0。

    请按顺序判断以下每个句子：

    """
            # 添加句子编号时保持原始编号
            for i, sentence in enumerate(chunk_sentences, start=start_idx + 1):
                chunk_prompt += f"句子{i}: {sentence['text']}\n"
                
            logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} (sentences {start_idx + 1} to {end_idx})")
            
            # 处理当前块
            chunk_labels, chunk_tokens = self._process_single_chunk(chunk_prompt)
            if chunk_labels:
                all_labels.extend(chunk_labels)
                # 累加token使用量
                total_tokens["input_tokens"] += chunk_tokens["input_tokens"]
                total_tokens["output_tokens"] += chunk_tokens["output_tokens"]
                total_tokens["total_tokens"] += chunk_tokens["total_tokens"]
                
                # 在处理下一个块之前休息一下
                if chunk_idx < num_chunks - 1:
                    time.sleep(20)
            else:
                logging.error(f"Failed to process chunk {chunk_idx + 1}")
                return [], total_tokens
        
        return all_labels, total_tokens    

    def _process_single_chunk(self, prompt: str) -> tuple[List[int], Dict]:
        """处理单个文本块，返回预测标签和token使用统计
        
        Args:
            prompt: 提示词文本
            
        Returns:
            tuple: (预测标签列表, token使用统计字典)
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": """你是一个专门用于识别学术论文中数据集描述的助手。
    你的主要任务是准确识别描述研究特定数据集的句子。
    判断原则：只有直接描述本研究所构建或使用的具体数据集的句子才标记为1,其他所有句子均标记为0。"""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.15
                )
                
                # 从响应中获取token使用信息
                token_usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                # 记录token使用情况
                logging.info(f"Token usage - Input: {token_usage['input_tokens']}, "
                        f"Output: {token_usage['output_tokens']}, "
                        f"Total: {token_usage['total_tokens']}")
                
                content = response.choices[0].message.content
                logging.debug(f"Model raw response:\n{content}")
                
                labels = []
                pattern = r'句子\d+[,，]标记为([01])[。.]'
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    match = re.search(pattern, line)
                    if match:
                        try:
                            label = int(match.group(1))
                            labels.append(label)
                            logging.debug(f"Extracted label {label} from line: {line}")
                        except (ValueError, IndexError) as e:
                            logging.debug(f"Failed to parse line: {line}, error: {str(e)}")
                
                if labels:
                    return labels, token_usage
                
                logging.error("No labels found in response")
                if attempt < self.max_retries - 1:
                    continue
                    
            except Exception as e:
                logging.error(f"API call error: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(300, (2 ** attempt))
                    logging.info(f"Attempt {attempt + 1} failed. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                raise
        
        return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def process_single_paper(self, input_file_path: str, paper_name: str) -> None:
        """
        处理单篇论文并更新其标签
        
        Args:
            input_file_path: JSON文件路径
            paper_name: 要处理的论文名称
        """
        try:
            # 读取JSON文件
            with open(input_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 找到指定的论文
            paper = None
            for item in json_data['data']:
                if item['paper_name'] == paper_name:
                    paper = item
                    break
            
            if not paper:
                logging.error(f"Paper not found: {paper_name}")
                return
            
            # 获取模型预测
            prompt = self.format_paper_for_prompt(paper)
            predicted_labels, token_usage = self.get_model_predictions(paper, prompt)
            
            if not predicted_labels:
                logging.error("Failed to get predictions")
                return
            
            # 检查预测标签数量
            if len(predicted_labels) != len(paper['sentences']):
                logging.error(
                    f"Label count mismatch: Expected {len(paper['sentences'])}, "
                    f"got {len(predicted_labels)}"
                )
                return
            
            # 更新标签
            paper_index = json_data['data'].index(paper)
            for i, sentence in enumerate(paper['sentences']):
                json_data['data'][paper_index]['sentences'][i]['label'] = predicted_labels[i]
            
            # 保存更新后的JSON
            with open(input_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            
            print(f"\n=== 处理完成: {paper_name} ===")
            print(f"总句子数: {len(paper['sentences'])}")
            print(f"标注为1的句子数: {sum(predicted_labels)}")
            print(f"Token使用情况:")
            print(f"  输入tokens: {token_usage['input_tokens']}")
            print(f"  输出tokens: {token_usage['output_tokens']}")
            print(f"  总tokens: {token_usage['total_tokens']}")
            
        except Exception as e:
            logging.error(f"Error processing paper: {str(e)}")

def extract_dataset_links(paper_text):
    """
    从论文文本中提取数据集链接
    整合自pdf2link.py的功能
    """
    client = OpenAI(
        api_key='sk-734cdbb2ddb94134a83041a2dcf863cf',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    prompt = f'''请从以下论文文本中提取该论文为了该研究而专门制作的数据集的URL，包括：

    论文全文：
    {paper_text}

    请按以下步骤分析论文：

    第一步：识别论文信息
    1. 确定输入论文的标题是什么
    2. 确定该论文的第一作者(first author)是谁
    3. 该论文发表于哪个时间

    第二步：理解研究内容
    1. 这篇论文的主要研究工作是什么
    2. 作者为了完成这项研究，创建了什么原创数据集
    3. 论文作者是否开源了项目代码？如果开源，在论文中的哪里提到了代码仓库链接

    第三步：提取关键链接
    请仅提取以下两类链接：

    1. 数据集链接 - 必须同时满足：
    - 该数据集是论文作者首次在本论文中提出的
    - 是为本研究专门创建的数据集
    - 论文中明确说明这是新的(new)或首次发布的(introduce/propose/present)数据集
    - 提供了明确的下载链接

    2. 官方代码仓库链接 - 必须满足：
    - 是论文作者开源的本项目的官方代码实现
    - 通常在论文中用"our code"、"source code"、"implementation"、"github"等词语引入
    - 位于论文正文、脚注或项目链接章节

    重要提示：
    - 请确保每个链接只返回一次，不要在dataset和code类别中包含重复相同的链接，请对最终的结果进行检查
    - 如果一个链接既包含数据集又包含代码，请只将其归类到一个最合适的类别中
    - 请仔细检查链接的有效性，确保它们是完整的URL

    请以JSON格式返回找到的链接:
    {{
        "urls": {{
            "dataset": ["数据集链接1", "数据集链接2", ...],
            "code": ["代码仓库链接1", "代码仓库链接2", ...]
        }}
    }}

    并说明为什么你认为返回的链接确实是输入论文的第一作者为该研究专门创建的数据集链接或者开源的代码链接。'''
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in extracting dataset URLs from academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 添加低temperature参数使输出更确定性
        )
        
        content = response.choices[0].message.content
        print(f"API响应内容: {content}")
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                
                # 提取所有链接
                all_urls = []
                
                # 从新的JSON结构中提取链接
                urls = result.get("urls", {})
                dataset_urls = urls.get("dataset", [])
                code_urls = urls.get("code", [])
                
                # 合并所有链接并去重
                all_urls = list(set(dataset_urls + code_urls))                      
                
                if all_urls:
                    print("\n找到的链接:")
                    if dataset_urls:
                        print("数据集链接:", dataset_urls)
                    if code_urls:
                        print("代码仓库链接:", code_urls)
                    
                return all_urls
            else:
                print("未在响应中找到JSON格式内容")
                return []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"尝试解析的JSON字符串: {json_str if 'json_str' in locals() else 'None'}")
            return []
            
    except Exception as e:
        print(f"API调用错误: {e}")
        print(f"完整错误信息: {str(e)}")
        return []

def process_dataset_links(links, download_dir, paper_name):
    """
    处理数据集链接，尝试下载数据集
    整合自agentest.py的功能
    """
    # 确保环境变量已设置
    if not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = "sk-734cdbb2ddb94134a83041a2dcf863cf"    
    results = []
    url_queue = queue.Queue()
    
    # 将所有链接加入队列
    for link in links:
        url_queue.put(link)
    
    # 设置最大处理轮数
    max_turns = 5
    cur_turn = 0
    history_dialogue = ''
    
    while cur_turn < max_turns and not url_queue.empty():
        cur_turn += 1
        # 获取下一个URL
        page_url = url_queue.get()
        print(f"\n处理链接 {cur_turn}/{max_turns}: {page_url}")
        
        # 检查是否可以直接下载
        if is_directlydown(page_url):
            folder_name = f'{paper_name}_dataset_{cur_turn}'
            download_path = os.path.join(download_dir, folder_name)
            os.makedirs(download_path, exist_ok=True)
            
            print(f"尝试直接下载: {page_url}")
            if download_dataset(page_url, folder_name, download_path):
                print(f"成功下载数据集到: {download_path}")
                results.append({
                    "url": page_url,
                    "status": "success",
                    "method": "direct_download",
                    "result": page_url  # 直接下载的链接作为结果
                })
                
                # 记录对话历史
                cur_dialogue = {
                    "thought": f'网页{page_url}可直接下载，故调用工具download_url直接下载该数据集',
                    'action': f'chosen download_url tool,download the url',
                    'observation': "download successfully"
                }
                history_dialogue += f"\n<step {cur_turn}> {json.dumps(cur_dialogue)}"
            else:
                print(f"下载失败: {page_url}")
                results.append({
                    "url": page_url,
                    "status": "failed",
                    "method": "direct_download",
                    "result": ""
                })
        else:
            # 需要分析网页
            print(f"需要分析网页: {page_url}")
            
            # 确定页面类型
            page_name = ''
            if 'github.com' in page_url:
                page_name = 'github'
            elif 'huggingface' in page_url:
                page_name = 'huggingface'
            
            # 获取网页内容
            web_information = getHtml(page_url)
            
            # 构建提示词
            paper_abstract = "这是一篇关于数据集的论文"  # 可以从paper_text中提取
            test_sysprompt3 = sysprompt3.format(page_name=page_name, abstract=paper_abstract, page_url=page_url)
            modelprompt = test_sysprompt3 + web_information + tool_prompt + history_dialogue + caution_prompt_o1
            
            try:
                # 调用模型分析网页
                model_response = Chat(message=modelprompt, model_name='qwen-plus')
                
                if model_response:
                    json_result = extract_final_json(model_response)
                    if json_result:
                        # 创建基础结果字典
                        current_result = {
                            "url": page_url,
                            "status": "analyzed",
                            "method": "web_analysis",
                            "result": ""  # 初始化为空字符串
                        }
                        
                        # 处理模型返回的结果
                        if json_result.get("next_action") == "final_answer":
                            if json_result.get("tool_name") == "candidate_urls":
                                url_list = json_result.get("request_arguments", {}).get("url_list", [])
                                # 只保留数据集链接
                                dataset_links = [url for url in url_list if any(platform in url.lower() for platform in 
                                               ['kaggle.com/datasets', 'data.mendeley.com', 'zenodo.org', 'figshare.com'])]
                                if dataset_links:
                                    current_result["result"] = dataset_links[0]  # 保存第一个有效的数据集链接
                                    
                                # 将新的数据集URL添加到队列
                                for url in dataset_links[1:]:  # 跳过第一个已保存的链接
                                    if url not in [r.get("url") for r in results]:
                                        url_queue.put(url)
                            
                        elif json_result.get("next_action") == "candidate_urls":
                            url_list = json_result.get("request_arguments", [])
                            dataset_links = [url for url in url_list if any(platform in url.lower() for platform in 
                                           ['kaggle.com/datasets', 'data.mendeley.com', 'zenodo.org', 'figshare.com'])]
                            if dataset_links:
                                current_result["result"] = dataset_links[0]
                                
                                # 将新的数据集URL添加到队列
                                for url in dataset_links[1:]:
                                    if url not in [r.get("url") for r in results]:
                                        url_queue.put(url)
                            
                        elif json_result.get("next_action") == "download_url":
                            download_url = json_result.get("request_arguments", "")
                            if download_url:
                                current_result["result"] = download_url
                        
                        # 添加处理结果
                        results.append(current_result)
                        
                    else:
                        results.append({
                            "url": page_url,
                            "status": "failed",
                            "method": "web_analysis",
                            "result": ""
                        })
                        print("无法从模型响应中提取JSON结果")
                else:
                    results.append({
                        "url": page_url,
                        "status": "failed",
                        "method": "web_analysis",
                        "result": ""
                    })
                    print("模型未返回响应")
                    
            except Exception as e:
                results.append({
                    "url": page_url,
                    "status": "failed",
                    "method": "web_analysis",
                    "result": str(e)
                })
                print(f"分析网页时出错: {str(e)}")
                
        # 每次处理完一个URL后休息一下
        if cur_turn < max_turns and not url_queue.empty():
            sleep_time = 10
            print(f"休息 {sleep_time} 秒后继续...")
            time.sleep(sleep_time)
    
    return results

def update_json_with_download_results(json_file_path, download_results):
    """
    将下载结果添加到JSON文件
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 添加下载结果
        data["dataset_download_results"] = download_results
        
        # 保存更新后的JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"已更新JSON文件: {json_file_path}")
    except Exception as e:
        print(f"更新JSON文件时出错: {str(e)}")


def dataset_description_and_link_extraction():
    # **************************************************
    # 这个地方后面记得修改过来
    today = get_date(2025, 9, 12) 
    print(f"Today is {today}")
    
    
    # 1. 首先下载pdf,需要提取出有数据集的论文的url 
    for category in categories:
        print(f"Now category is {category}")
        file_path = f"/home/kemove/project/DC/output/2_paper_processed/{today}/process_{category}.json"
        values = extract_json_values(file_path, "Sample") # 提取有数据集的id（pre处理后），将返回值赋给 values 变量!
        dataset_file_path = f"/home/kemove/project/DC/output/1_paper/{today}/{category}.json"
        # 将找到的values放进去，进行匹配，从而找到每一小类的pdf的url
        pdf_urls = find_json_url(dataset_file_path, values, category)
        pdf_save_dir = f"/home/kemove/project/DC/output/3_pdf/{today}/{category}"  # PDF保存目录
        
        # 将有自建数据集的论文下载下来
        if pdf_urls:
            # 不需要在这里创建文件夹了，download_pdf 函数会处理
            for i, url in enumerate(pdf_urls):
                print(f"Now url is {url}")
                filename = f"{category}_{values[i]}.pdf"  # 生成唯一文件名
                save_path = f"{pdf_save_dir}/{filename}"  # 完整的保存路径 (目录 + 文件名)
                
                # 下载和处理pdf,并生成txt文件
                download_pdf(url, save_path)  # 传入完整的路径 
                time.sleep(5) # 每次下载完休息一下
                #2.处理pdf，切分成doc，写入txt（仅含一个标题）
                # print(f"Now save_path is {save_path}")
                test_pdf_processing(pdf_save_dir, filename)
                
                txt_filename = filename.replace('.pdf', '.txt')
                json_name = f"{category}_{values[i]}.json"
                json_path = f"{pdf_save_dir}/{json_name}"
                
                # 创建和处理json，将处理好的txt文件存入到json中
                if not os.path.exists(json_path):
                    create_new_json(pdf_save_dir, json_name)

                #3.为每一行加标题：处理txt文件便于后续存入到json中进行预处理
                process_single_file(pdf_save_dir, pdf_save_dir, txt_filename)

                #4.处理txt文件并且将其转化为特定的json格式
                process_txt_to_json(f"{pdf_save_dir}/{txt_filename}", json_path)

                
                print("*******************OVER*******************")
                # 使用大模型进行标注
                evaluator = ResearchDatasetEvaluator()
                with open(json_path, 'r', encoding='utf-8') as f:
                    print(f"Now is reading {json_name} file")
                    json_str = f.read()      
                    
                # 5.使用大模型进行标注数据集描述获取标注结果
                paper_data = evaluator.load_data(json_str)
                # 
                if paper_data and len(paper_data) > 0:
                    paper_name = paper_data[0]['paper_name']
                    evaluator.process_single_paper(json_path, paper_name)
                    print(f"已处理并更新 {json_name} 文件")
                    # 6.使用大模型提取数据集链接
                    print("\n开始提取数据集链接...")
                    # 6.1. 从txt文件中读取论文文本
                    with open(f"{pdf_save_dir}/{txt_filename}", 'r', encoding='utf-8') as f:
                        paper_text = f.read()
                    # 6.2. 提取数据集链接
                    dataset_links = extract_dataset_links(paper_text)
                    if dataset_links:
                        print(f"找到 {len(dataset_links)} 个候选数据集链接")
                        # 6.3. 处理和尝试下载数据集
                        download_dir = f"/arxiv-dataset/dataset_stroage/{today}/{category}"
                        download_results = process_dataset_links(dataset_links, download_dir, f"{category}_{values[i]}")
                        # 6.4. 将下载结果添加到JSON文件
                        update_json_with_download_results(json_path, download_results)
                    else:
                        print("未找到数据集链接")    
                else:
                    print(f"没有找到任何数据集描述句子")
        else:
            print(f"这个小类没有包含自建数据集的论文！")



dataset_description_and_link_extraction()


