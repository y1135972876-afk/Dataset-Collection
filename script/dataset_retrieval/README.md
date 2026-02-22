# 数据集检索框架

这是一个可扩展的数据集检索框架，支持with instruction和without instruction两种检索模式。框架基于Alibaba-NLP/gte-large-en-v1.5模型实现，并支持后续扩展到其他检索模型。

## 功能特点

- **单命令实验**: 只需指定模型名称与是否拼接指令即可运行完整批量实验
- **整文档索引**: GTE 将数据集中每条记录的全部字段作为检索文本进行编码
- **完整评估指标**: 输出 Hit Rate / MRR / Precision / Recall / F1 / NDCG / Average Precision 等核心检索指标
- **结果持久化**: 自动保存实验信息与详细评估结果为 JSON 文件
- **可扩展设计**: 易于添加新的检索模型或替换现有模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据格式

### 查询数据 (query.json)
每个查询包含以下字段：
- `query`: 用户查询文本
- `instruction`: 检索指令（可选）
- `positives`: 正例链接列表
- `ground_truth`: 标准答案信息
- `metadata`: 元数据信息

### 数据集数据 (final_dataset.json)
每个数据集包含以下字段：
- `Paper Link`: 论文链接
- `Dataset Link`: 数据集链接
- `Dataset Description`: 数据集描述
- `Timestamp`: 时间戳

检索阶段会将每条数据集记录的全部字段拼接成单个文档传入模型，无需手动抽取描述或链接。

## 使用方法

运行完整批量评估：

```bash
# 默认使用 Alibaba-NLP/gte-large-en-v1.5，并且不拼接指令
python main.py

# 指定其他模型
python main.py --model IntelliTech/my-gte-checkpoint

# 使用 instruction
python main.py --with-instruction
```

## 命令行参数

- `--model`: 指定用于检索的模型名称（默认: `Alibaba-NLP/gte-large-en-v1.5`）
- `--with-instruction`: 是否在查询中拼接instruction信息

## 输出结果

实验结果保存在`results/`目录下：
- `{experiment_name}/result.json`: 每条查询的 Top-10 排名与分数
- `{experiment_name}/score.json`: 各指标的平均得分（如 `ndcg@1`、`recall@5`）

## 扩展新的检索模型

1. 在`retriever/`目录下创建新的检索器类，继承`BaseRetriever`
2. 在`RetrievalFramework.initialize_retriever()`中添加新的模型类型
3. 实现必要的抽象方法：
   - `initialize()`: 初始化模型
   - `encode_texts()`: 批量编码文本
   - `encode_text()`: 单个文本编码
   - `_extract_search_text()`: 提取检索文本

## 评估指标

- **Hit Rate@k**: 在top-k结果中是否找到任一相关文档
- **MRR@k**: Mean Reciprocal Rank，相关文档的倒数排名
- **Precision@k / Recall@k / F1@k**
- **NDCG@k**: 归一化折损累积增益
- **Average Precision@k**

## 示例输出

```
==================================================
检索评估结果摘要
==================================================
总查询数: 63
有效查询数: 63

各指标结果:
Top-1:
  Valid Queries: 63
  average_precision: 0.776 ± 0.037
  f1: 0.721 ± 0.038
  hit_rate: 0.857 ± 0.034
  mrr: 0.857 ± 0.034
  ndcg: 0.857 ± 0.034
  precision: 0.857 ± 0.034
  recall: 0.621 ± 0.042
Top-5:
  Valid Queries: 63
  average_precision: 0.864 ± 0.028
  f1: 0.559 ± 0.022
  hit_rate: 0.968 ± 0.020
  mrr: 0.898 ± 0.030
  ndcg: 0.921 ± 0.024
  precision: 0.412 ± 0.018
  recall: 0.893 ± 0.025
```

## 架构说明

- `retriever/base_retriever.py`: 检索器基类
- `retriever/gte_retriever.py`: GTE模型检索器实现
- `utils/data_loader.py`: 数据加载器
- `utils/evaluator.py`: 评估器
- `retrieval_framework.py`: 主框架类
- `main.py`: 命令行入口
