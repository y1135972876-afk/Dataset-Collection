#!/usr/bin/env python3
"""数据集检索框架主程序"""

import argparse
from pathlib import Path
from retrieval_framework import RetrievalFramework


def main():
    """执行完整的检索评估流程"""
    parser = argparse.ArgumentParser(description="数据集检索评估")
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-large-en-v1.5",
        help="指定用于检索的模型名称"
    )
    parser.add_argument(
        "--with-instruction",
        action="store_true",
        help="是否在查询中拼接指令信息"
    )
    args = parser.parse_args()

    print("初始化检索框架...")
    framework = RetrievalFramework(data_dir="datasets")

    stats = framework.get_data_statistics()
    print(f"数据统计: {stats['num_queries']} 个查询, {stats['num_datasets']} 个数据集")

    framework.initialize_retriever("gte", model_name=args.model)

    results = framework.run_experiment(
        with_instruction=args.with_instruction,
        top_k=10,
        top_k_values=[1, 5, 10],
        save_results=True,
        output_dir="results"
    )

    experiment_name = results["experiment_info"]["experiment_name"]
    experiment_dir = Path("results") / experiment_name
    print(f"\n完整评估已完成。")
    print(f"Top-10 排名及分数: {experiment_dir / 'result.json'}")
    print(f"平均指标得分: {experiment_dir / 'score.json'}")


if __name__ == "__main__":
    main()
