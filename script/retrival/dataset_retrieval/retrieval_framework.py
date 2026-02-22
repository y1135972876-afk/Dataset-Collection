from typing import List, Dict, Any, Optional, Tuple, Union
import time
from pathlib import Path
from retriever.base_retriever import BaseRetriever
from retriever.gte_retriever import GTERetriever
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator


class RetrievalFramework:
    """数据集检索框架主类"""

    def __init__(self, data_dir: str = ""):
        self.data_loader = DataLoader(data_dir)
        self.evaluator = Evaluator()
        self.retrievers = {}  # 存储不同类型的检索器
        self.current_retriever = None
        self.use_faiss = False

    def initialize_retriever(self, retriever_type: str = "gte",
                             model_name: str = "Alibaba-NLP/gte-large-en-v1.5",
                             device: str = None,
                             use_faiss: bool = False) -> None:
        """
        初始化检索器

        Args:
            retriever_type: 检索器类型 ("gte")
            model_name: 模型名称
            device: 设备 ("cuda", "cpu", 或None自动选择)
            use_faiss: 是否启用Faiss索引
        """
        if retriever_type == "gte":
            retriever = GTERetriever(model_name, device)
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        print(f"Initializing {retriever_type} retriever...")
        start_time = time.time()
        retriever.initialize()
        retriever.configure_indexing(use_faiss=use_faiss)
        init_time = time.time() - start_time

        self.retrievers[retriever_type] = retriever
        self.current_retriever = retriever
        self.use_faiss = retriever.is_faiss_enabled()

        faiss_msg = "enabled" if self.use_faiss else "disabled"
        print(f"{retriever_type} retriever initialized in {init_time:.2f}s (Faiss {faiss_msg}).")

    def retrieve_single_query(self, query: Dict[str, Any],
                            with_instruction: bool = False,
                            top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        对单个查询进行检索

        Args:
            query: 查询字典
            with_instruction: 是否使用instruction
            top_k: 返回前k个结果

        Returns:
            检索结果列表 (dataset, score)
        """
        if self.current_retriever is None:
            raise RuntimeError("No retriever initialized. Call initialize_retriever() first.")

        # 获取查询文本
        query_text = self.data_loader.get_query_text(query)

        # 获取指令（如果需要）
        instruction = None
        if with_instruction:
            instruction = self.data_loader.get_instruction(query)

        # 获取候选数据集
        candidates = self.data_loader.get_datasets()

        # 执行检索
        results = self.current_retriever.retrieve(
            query=query_text,
            candidates=candidates,
            instruction=instruction,
            top_k=top_k,
            with_instruction=with_instruction,
            use_faiss=self.use_faiss
        )

        return results

    def retrieve_batch(self, queries: List[Dict[str, Any]],
                      with_instruction: bool = False,
                      top_k: int = 10) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        批量检索多个查询

        Args:
            queries: 查询列表
            with_instruction: 是否使用instruction
            top_k: 返回前k个结果

        Returns:
            检索结果批次
        """
        results = []
        for i, query in enumerate(queries):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(queries)} queries")

            result = self.retrieve_single_query(query, with_instruction, top_k)
            results.append(result)

        return results

    def evaluate_single_query(self, query: Dict[str, Any],
                            retrieval_results: List[Tuple[Dict[str, Any], float]],
                            top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        评估单个查询的检索结果
        """
        return self.evaluator.evaluate_retrieval(query, retrieval_results, top_k_values)

    def evaluate_batch(self, queries: List[Dict[str, Any]],
                      retrieval_results_batch: List[List[Tuple[Dict[str, Any], float]]],
                      top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        批量评估检索结果
        """
        return self.evaluator.evaluate_batch(queries, retrieval_results_batch, top_k_values)

    def run_experiment(self, with_instruction: bool = False,
                      top_k: int = 10, top_k_values: List[int] = [1, 3, 5, 10],
                      save_results: bool = True, output_dir: str = "results") -> Dict[str, Any]:
        """
        运行完整的检索实验

        Args:
            with_instruction: 是否使用instruction
            top_k: 检索时返回的top-k值
            top_k_values: 评估时使用的top-k值列表
            save_results: 是否保存结果
            output_dir: 输出目录

        Returns:
            实验结果字典
        """
        experiment_name = self._format_experiment_name(with_instruction)
        print(f"Running experiment: {experiment_name}")

        # 加载数据
        queries = self.data_loader.get_queries()

        # 执行检索
        print("Starting retrieval...")
        start_time = time.time()
        retrieval_results = self.retrieve_batch(queries, with_instruction, top_k)
        retrieval_time = time.time() - start_time
        avg_time = retrieval_time / len(queries) if queries else 0.0
        print(f"Retrieval finished in {retrieval_time:.2f}s "
              f"(avg {avg_time:.2f}s per query).")

        # 评估结果
        print("Evaluating results...")
        evaluation_results = self.evaluate_batch(queries, retrieval_results, top_k_values)

        # 添加实验信息
        experiment_info = {
            "experiment_name": experiment_name,
            "with_instruction": with_instruction,
            "retriever_type": type(self.current_retriever).__name__,
            "model_name": self.current_retriever.model_name,
            "model_slug": self._slugify_model_name(self.current_retriever.model_name),
            "retrieval_top_k": top_k,
            "evaluation_top_k_values": top_k_values,
            "total_queries": len(queries),
            "retrieval_time_seconds": retrieval_time,
            "avg_time_per_query": avg_time
        }

        serialized_retrievals = self._serialize_retrieval_results(
            queries=queries,
            retrieval_results=retrieval_results,
            top_k=top_k
        )

        results = {
            "experiment_info": experiment_info,
            "evaluation_results": evaluation_results,
            "retrieval_rankings": serialized_retrievals
        }

        # 保存结果
        if save_results:
            self._save_experiment_results(results, output_dir)

        # 打印摘要
        self.evaluator.print_evaluation_summary(evaluation_results)

        return results

    def compare_experiments(self, experiments: Dict[str, Dict[str, Any]],
                          top_k: int = 5, output_dir: str = "results") -> None:
        """
        比较多个实验的结果

        Args:
            experiments: 实验名称到实验结果的映射
            top_k: 比较的top-k值
            output_dir: 输出目录
        """
        print("\nComparing experiments...")

        # 提取评估结果
        method_results = {}
        for exp_name, exp_results in experiments.items():
            method_results[exp_name] = exp_results["evaluation_results"]

        # 使用评估器进行比较
        self.evaluator.compare_methods(method_results, top_k)

        # 保存比较结果
        comparison_file = Path(output_dir) / "experiment_comparison.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)

        with open(comparison_file, 'w', encoding='utf-8') as f:
            import json
            json.dump({
                "experiments": list(experiments.keys()),
                "comparison_top_k": top_k,
                "results": method_results
            }, f, indent=2, ensure_ascii=False)

        print(f"Comparison results saved to: {comparison_file}")

    def _save_experiment_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """保存实验结果"""
        output_path = Path(output_dir)
        experiment_info = results["experiment_info"]
        experiment_name = experiment_info["experiment_name"]

        experiment_dir = output_path / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        import json

        # 保存检索排名
        result_file = experiment_dir / "result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_name": experiment_name,
                "model_name": experiment_info["model_name"],
                "with_instruction": experiment_info["with_instruction"],
                "retrieval_top_k": experiment_info["retrieval_top_k"],
                "results": results["retrieval_rankings"]
            }, f, indent=2, ensure_ascii=False)

        # 保存平均得分
        score_file = experiment_dir / "score.json"
        aggregated = results["evaluation_results"].get("aggregated_metrics", {})
        averaged_metrics = {
            metric_key: stats.get("mean", 0.0)
            for metric_key, stats in aggregated.items()
        }
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_name": experiment_name,
                "model_name": experiment_info["model_name"],
                "with_instruction": experiment_info["with_instruction"],
                "metrics": averaged_metrics
            }, f, indent=2, ensure_ascii=False)

        print(f"Ranking results saved to: {result_file}")
        print(f"Average scores saved to: {score_file}")

    def get_retriever_info(self) -> Dict[str, Any]:
        """获取当前检索器信息"""
        if self.current_retriever is None:
            return {"error": "No retriever initialized"}

        if hasattr(self.current_retriever, 'get_model_info'):
            return self.current_retriever.get_model_info()
        else:
            return {
                "model_name": self.current_retriever.model_name,
                "is_initialized": self.current_retriever.is_initialized
            }

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        return self.data_loader.get_statistics()

    def _slugify_model_name(self, model_name: str) -> str:
        """将模型名称转换为适合文件名的格式"""
        allowed_chars = "-_."
        slug = "".join(
            ch if ch.isalnum() or ch in allowed_chars else "_"
            for ch in model_name
        )
        return slug.strip("_") or "model"

    def _format_experiment_name(self, with_instruction: bool) -> str:
        """生成包含模型信息的实验名称"""
        instruction_part = "with_instruction" if with_instruction else "without_instruction"
        return f"{self._slugify_model_name(self.current_retriever.model_name)}_{instruction_part}"

    def _serialize_retrieval_results(self,
                                     queries: List[Dict[str, Any]],
                                     retrieval_results: List[List[Tuple[Dict[str, Any], float]]],
                                     top_k: int) -> List[Dict[str, Any]]:
        """整理检索结果用于保存"""
        serialized: List[Dict[str, Any]] = []
        for idx, (query, results) in enumerate(zip(queries, retrieval_results)):
            query_id = (
                query.get("metadata", {}).get("generated_at")
                or query.get("id")
                or f"query_{idx}"
            )
            top_entries = []
            for rank, (dataset, score) in enumerate(results[:top_k], start=1):
                top_entries.append({
                    "rank": rank,
                    "score": float(score),
                    "dataset_link": dataset.get("Dataset Link", ""),
                    "paper_link": dataset.get("Paper Link", ""),
                    "dataset_description": dataset.get("Dataset Description", "")
                })
            serialized.append({
                "query_id": query_id,
                "query": query.get("query", ""),
                "instruction": query.get("instruction", ""),
                "top_results": top_entries
            })
        return serialized
