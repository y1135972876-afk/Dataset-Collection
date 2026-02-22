import json
from typing import List, Dict
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
from collections import defaultdict
from itertools import groupby
import logging
import sys
import os
from datetime import datetime
import time

# 设置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class ResearchDatasetEvaluator:
    def __init__(self):
        logging.info("Initializing ResearchDatasetEvaluator")
        self.client = OpenAI(
            api_key='sk-734cdbb2ddb94134a83041a2dcf863cf',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.max_retries = 5
        self.chunk_size = 8000
        
        # 设置结果文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_{timestamp}"
        self.papers_dir = os.path.join(self.results_dir, "papers")
        self.final_results_file = os.path.join(self.results_dir, "final_evaluation_results.json")
        self.interim_results_file = os.path.join(self.results_dir, "interim_results.json")
        
        # 创建必要的目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.papers_dir, exist_ok=True)
        
    def calculate_metrics(self, true_labels: List[int], predicted_labels: List[int]) -> Dict:
        """计算单篇文章的评估指标"""
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, zero_division=0),
            "recall": recall_score(true_labels, predicted_labels, zero_division=0),
            "f1": f1_score(true_labels, predicted_labels, zero_division=0),
            "classification_report": classification_report(true_labels, predicted_labels, 
                                                        output_dict=True, zero_division=0)
        }        


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
        formatted_text = f"""你是一位专业的学术论文数据集描述识别专家。请仔细分析下面这篇论文中的每个句子，判断它们是否描述了本研究特定构建的数据集信息。

    论文标题: {paper['paper_name']}

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
    
    请对每个句子进行分析，返回一个JSON格式的标签数组。格式要求：
    1. 必须是有效的JSON格式
    2. 只包含labels字段，值为0和1组成的数组
    3. 数组长度必须与句子数量相同
    4. 示例格式：{{"labels": [0,1,0,1,...]}}

    需要判断的句子：

    """
        for i, sentence in enumerate(paper['sentences'], 1):
            formatted_text += f"{i}. {sentence['text']}\n"
        
        formatted_text += "\n请严格按照JSON格式返回：{\"labels\": [0,1,0,...]}，不要包含任何其他文字。"
        
        return formatted_text

    def evaluate_paper(self, paper: Dict, paper_idx: int, total_papers: int) -> Dict:
        """评估单篇论文并保存结果"""
        logging.info(f"Evaluating paper {paper_idx}/{total_papers}: {paper['paper_name']}")
        
        # 获取模型预测
        prompt = self.format_paper_for_prompt(paper)
        predicted_labels = self.get_model_predictions(prompt)
        
        if not predicted_labels:
            logging.warning(f"No predictions obtained for paper: {paper['paper_name']}")
            return None
            
        true_labels = [sentence['label'] for sentence in paper['sentences']]
        
        if len(predicted_labels) != len(true_labels):
            logging.error(f"Label count mismatch for {paper['paper_name']}")
            return None
        
        # 计算评估指标
        metrics = self.calculate_metrics(true_labels, predicted_labels)
        
        # 分析预测差异
        differences = self.analyze_prediction_differences(
            paper_name=paper['paper_name'],
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            sentences=paper['sentences']
        )
        
        # 准备完整的论文评估结果
        paper_result = {
            "paper_name": paper['paper_name'],
            "paper_index": paper_idx,
            "total_papers": total_papers,
            "metrics": metrics,
            "differences": differences,
            "raw_data": {
                "true_labels": true_labels,
                "predicted_labels": predicted_labels
            }
        }
        
        # 保存当前论文结果
        # self.save_paper_result(paper_result, paper_idx)
        self.save_results(paper_result, "paper", paper_idx, total_papers)
        
        # 打印当前论文的评估结果
        print(f"\n=== 论文评估结果 ({paper_idx}/{total_papers}): {paper['paper_name']} ===")
        print(f"总句子数: {len(true_labels)}")
        print(f"预测错误句子数: {differences['total_differences']}")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        
        if differences['total_differences'] > 0:
            print("\n预测错误的句子:")
            for diff in differences['differences']:
                print(f"\nSentence {diff['sentence_id']}:")
                print(f"文本: {diff['text']}")
                print(f"真实标签: {diff['true_label']}")
                print(f"预测标签: {diff['predicted_label']}")
                print(f"错误类型: {diff['error_type']}")
        
        return paper_result
    

    def analyze_prediction_differences(self, paper_name: str, true_labels: List[int], 
                                     predicted_labels: List[int], sentences: List[Dict]) -> Dict:
        """分析预测标签与真实标签的差异"""
        differences = []
        for idx, (true, pred, sentence) in enumerate(zip(true_labels, predicted_labels, sentences)):
            if true != pred:
                differences.append({
                    "sentence_id": idx + 1,
                    "text": sentence["text"],
                    "true_label": true,
                    "predicted_label": pred,
                    "error_type": "False Positive" if pred == 1 else "False Negative"
                })
        
        return {
            "paper_name": paper_name,
            "total_sentences": len(true_labels),
            "total_differences": len(differences),
            "differences": differences
        }

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

    def get_model_predictions(self, prompt: str) -> List[int]:
        """使用Qwen-Plus模型获取预测标签"""
        logging.info("Starting model prediction")
        
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Attempt {attempt + 1} of {self.max_retries}")
                start_time = time.time()
                
                # 修改调用格式
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {
                            "role": "system",
                            "content": """你是一个专门用于识别学术论文中数据集描述的助手。你需要判断每个输入的句子是否描述了研究中使用的数据集。请确保：
    1. 只输出JSON格式的结果
    2. 结果格式必须为 {"labels": [0,1,0,...]}
    3. 不要输出任何其他文字或解释"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"}  # 指定返回格式为JSON
                )
                
                elapsed_time = time.time() - start_time
                logging.debug(f"API request completed in {elapsed_time:.2f} seconds")
                
                content = response.choices[0].message.content.strip()
                logging.debug(f"Raw model response: {content}")
                
                try:
                    result = json.loads(content)
                    labels = result.get("labels", [])
                    
                    if not isinstance(labels, list):
                        logging.error("Invalid label format: not a list")
                        raise ValueError("模型返回的标签不是列表格式")
                    
                    labels = [1 if label == 1 else 0 for label in labels]
                    logging.info(f"Successfully processed {len(labels)} labels")
                    
                    return labels
                    
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error: {str(e)}")
                    logging.error(f"Problematic content: {content}")
                    continue
                    
            except Exception as e:
                logging.error(f"API call error on attempt {attempt + 1}: {str(e)}")
                # 添加更详细的错误日志
                if hasattr(e, 'response'):
                    logging.error(f"Response status: {e.response.status_code}")
                    logging.error(f"Response body: {e.response.text}")
                if attempt == self.max_retries - 1:
                    raise
                continue
        
        return []
    
    def _generate_comparison(self, sentences: List[Dict], predicted_labels: List[int]) -> List[Dict]:
        """生成预测结果与真实标签的对比"""
        if not sentences or not predicted_labels:
            return []
            
        true_labels = [sentence['label'] for sentence in sentences]
        
        return [
            {
                "sentence_id": idx + 1,
                "text": sentence["text"],
                "true_label": true,
                "predicted_label": pred,
                "is_correct": true == pred,
                "error_type": None if true == pred else ("False Positive" if pred == 1 else "False Negative")
            }
            for idx, (sentence, true, pred) in enumerate(zip(sentences, true_labels, predicted_labels))
        ]

    def _calculate_statistics(self, sentences: List[Dict], predicted_labels: List[int]) -> Dict:
        """计算预测统计信息"""
        if not sentences or not predicted_labels:
            return {}
            
        true_labels = [sentence['label'] for sentence in sentences]
        
        return {
            "total_sentences": len(sentences),
            "correct_predictions": sum(t == p for t, p in zip(true_labels, predicted_labels)),
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "false_positives": sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1),
            "false_negatives": sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
        }    

    def get_label_comparison(self, paper_name: str, true_labels: List[int], 
                            predicted_labels: List[int], sentences: List[Dict]) -> Dict:
        """分析标签预测结果与真实标签的对比"""
        comparison = {
            "paper_name": paper_name,
            "total_sentences": len(true_labels),
            "correct_predictions": sum(t == p for t, p in zip(true_labels, predicted_labels)),
            "details": []
        }
        
        for idx, (true, pred, sentence) in enumerate(zip(true_labels, predicted_labels, sentences)):
            comparison["details"].append({
                "sentence_id": idx + 1,
                "text": sentence["text"],
                "true_label": true,
                "predicted_label": pred,
                "correct": true == pred,
                "error_type": None if true == pred else ("False Positive" if pred == 1 else "False Negative")
            })
        
        return comparison

    def evaluate_all_papers(self, papers: List[Dict]) -> Dict:
        """评估所有论文并计算整体指标"""
        all_results = []
        all_true_labels = []
        all_predicted_labels = []
        
        total_papers = len(papers)
        logging.info(f"Starting evaluation of {total_papers} papers")
        
        for paper_idx, paper in enumerate(papers, 1):
            try:
                # 使用evaluate_paper方法评估单篇论文
                paper_result = self.evaluate_paper(paper, paper_idx, total_papers)
                
                if paper_result:
                    # 使用统一的save_results方法保存结果
                    self.save_results(paper_result, "paper", paper_idx, total_papers)
                    self.save_results(paper_result, "interim", paper_idx, total_papers)
                    
                    all_results.append(paper_result)
                    all_true_labels.extend(paper_result['raw_data']['true_labels'])
                    all_predicted_labels.extend(paper_result['raw_data']['predicted_labels'])
                    
            except Exception as e:
                logging.error(f"Error processing paper {paper['paper_name']}: {str(e)}")
                continue

        # 计算整体指标
        if all_true_labels and all_predicted_labels:
            overall_metrics = self.calculate_metrics(all_true_labels, all_predicted_labels)
            
            final_results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_papers": total_papers,
                "processed_papers": len(all_results),
                "overall_metrics": overall_metrics,
                "paper_results": all_results,
                "paper_results_directory": self.papers_dir
            }
            
            # 使用统一的save_results方法保存最终结果
            self.save_results(final_results, "final")
            
            # 打印整体评估结果
            print("\n=== 总体评估结果 ===")
            print(f"评估论文总数: {len(papers)}")
            print(f"总句子数: {len(all_true_labels)}")
            print(f"准确率: {overall_metrics['accuracy']:.4f}")
            print(f"精确率: {overall_metrics['precision']:.4f}")
            print(f"召回率: {overall_metrics['recall']:.4f}")
            print(f"F1分数: {overall_metrics['f1']:.4f}")
            
            return final_results
        else:
            logging.error("No valid results obtained from any paper")
            return None



def main():
    """主函数"""
    evaluator = ResearchDatasetEvaluator()
    
    input_file = '/home/kzlab/muse/Savvy/Data_collection/script/reconstruct_dataset/test_revise.json'
    print(f"正在读取数据文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        json_str = f.read()
    
    papers = evaluator.load_data(json_str)
    results = evaluator.evaluate_all_papers(papers)
    
    print("\n=== 总体评估结果 ===")
    print(f"评估论文总数: {results['overall_metrics']['total_papers']}")
    print(f"总句子数: {results['overall_metrics']['total_sentences']}")
    print(f"数据集描述句子总数: {results['overall_metrics']['total_dataset_sentences']}")
    print(f"整体准确率: {results['overall_metrics']['accuracy']:.4f}")
    print(f"整体精确率: {results['overall_metrics']['precision']:.4f}")
    print(f"整体召回率: {results['overall_metrics']['recall']:.4f}")
    print(f"整体F1分数: {results['overall_metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()