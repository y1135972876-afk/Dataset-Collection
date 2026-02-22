from typing import List, Dict, Any, Tuple, Set
import numpy as np
import math


class Evaluator:
    """检索结果评估器"""

    def __init__(self):
        self.metrics = {}

    def _extract_candidate_link(self, candidate: Dict[str, Any]) -> str:
        """
        从候选文档中抽取一个用于匹配的主链接。

        规则：
        1. 一定优先使用 Paper Link（arxiv PDF）
        2. 如果没有 Paper Link，再退回 Dataset Link / link
        """

        # 1) 首选：Paper Link
        for key in ("Paper Link", "paper_link"):
            v = candidate.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 2) 兜底：Dataset Link / link
        for key in ("Dataset Link", "dataset_link", "link"):
            v = candidate.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return ""


    def evaluate_retrieval(self, query: Dict[str, Any],
                          retrieval_results: List[Tuple[Dict[str, Any], float]],
                          top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        评估单个查询的检索结果

        Args:
            query: 查询数据
            retrieval_results: 检索结果列表 (dataset, score)
            top_k_values: 评估的top-k值

        Returns:
            评估结果字典
        """
        # 1) 收集 ground truth 链接集合（优先 paper_link / positives）
        relevant_links = self._get_relevant_links(query)
        if not relevant_links:
            return {"error": "No ground truth found"}

        # 2) 从候选中抽取“主链接”：一定优先 Paper Link，退化时才用 Dataset Link / link
        retrieved_links = [
            self._extract_candidate_link(dataset)
            for (dataset, score) in retrieval_results
        ]
        
        metrics_flat: Dict[str, float] = {}
        results = {
            "query_id": query.get("metadata", {}).get("generated_at", "unknown"),
            "ground_truth_links": sorted(relevant_links),
            "num_results": len(retrieval_results),
            "metrics": metrics_flat
        }

        for k in top_k_values:
            metrics = self._compute_metrics_at_k(relevant_links, retrieved_links, k)
            for metric_name, value in metrics.items():
                if metric_name == "k":
                    continue
                metrics_flat[f"{metric_name}@{k}"] = value

        return results

    def evaluate_batch(self, queries: List[Dict[str, Any]],
                       retrieval_results_batch: List[List[Tuple[Dict[str, Any], float]]],
                       top_k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        批量评估检索结果

        Args:
            queries: 查询列表
            retrieval_results_batch: 检索结果批次
            top_k_values: 评估的top-k值

        Returns:
            批量评估结果
        """
        if len(queries) != len(retrieval_results_batch):
            raise ValueError("Queries and results batch must have the same length")

        metric_names = ["hit_rate", "mrr", "precision", "recall", "f1", "ndcg", "average_precision"]
        metric_keys = [f"{metric}@{k}" for k in top_k_values for metric in metric_names]
        individual_results = []
        aggregated_values: Dict[str, List[float]] = {key: [] for key in metric_keys}

        for query, results in zip(queries, retrieval_results_batch):
            result = self.evaluate_retrieval(query, results, top_k_values)
            individual_results.append(result)

            # 收集聚合指标
            if "error" not in result:
                for key in metric_keys:
                    aggregated_values[key].append(result["metrics"].get(key, 0.0))

        # 计算平均值
        summary = {
            "num_queries": len(queries),
            "individual_results": individual_results,
            "aggregated_metrics": {}
        }

        aggregated_metrics = summary["aggregated_metrics"]
        for key, values in aggregated_values.items():
            aggregated_metrics[key] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "std": float(np.std(values)) if values else 0.0,
                "num_valid_queries": len(values)
            }

        return summary

    def _get_ground_truth_link(self, query: Dict[str, Any]) -> str:
        """从查询中提取ground truth链接"""
        ground_truth = query.get("ground_truth", {})
        return ground_truth.get("link", "")

    def _get_relevant_links(self, query: Dict[str, Any]) -> Set[str]:
        """
        获取该 query 的“正确链接”集合。

        设计原则：
        - 优先收集各种 paper_link / positives（通常是 arxiv PDF）
        - 兼容旧 schema：ground_truth.link / metadata.dataset_link 仍然保留
        - 兼容新 schema：ground_truth.paper_link / ground_truth.positives / meta.positives 等
        """
        relevant: Set[str] = set()

        gt = query.get("ground_truth")

        # 1) ground_truth 是 dict
        if isinstance(gt, dict):
            # 显式的 paper_link / dataset_link / link
            for key in ("paper_link", "dataset_link", "link"):
                v = gt.get(key)
                if isinstance(v, str) and v.strip():
                    relevant.add(v.strip())

            # ground_truth.positives 可以是 str 或 list
            pos = gt.get("positives")
            if isinstance(pos, str) and pos.strip():
                relevant.add(pos.strip())
            elif isinstance(pos, list):
                for v in pos:
                    if isinstance(v, str) and v.strip():
                        relevant.add(v.strip())

        # 2) ground_truth 是字符串（例如只给了数据集名）
        #    这时真正的链接一般在 meta / metadata 里
        elif isinstance(gt, str):
            pass  # 名字本身没用，下面从 meta 里找链接

        # 3) meta / metadata 里可能有 paper_link / dataset_link / positives
        for meta_key in ("meta", "metadata"):
            meta = query.get(meta_key)
            if isinstance(meta, dict):
                for key in ("paper_link", "dataset_link", "link"):
                    v = meta.get(key)
                    if isinstance(v, str) and v.strip():
                        relevant.add(v.strip())

                pos = meta.get("positives")
                if isinstance(pos, str) and pos.strip():
                    relevant.add(pos.strip())
                elif isinstance(pos, list):
                    for v in pos:
                        if isinstance(v, str) and v.strip():
                            relevant.add(v.strip())

        # 4) 顶层 positives（旧数据里有）
        pos = query.get("positives")
        if isinstance(pos, str) and pos.strip():
            relevant.add(pos.strip())
        elif isinstance(pos, list):
            for v in pos:
                if isinstance(v, str) and v.strip():
                    relevant.add(v.strip())

        # 5) 保留原始逻辑中的 metadata.dataset_link（完全向后兼容）
        meta_old = query.get("metadata")
        if isinstance(meta_old, dict):
            v = meta_old.get("dataset_link")
            if isinstance(v, str) and v.strip():
                relevant.add(v.strip())

        # 去掉空字符串
        relevant.discard("")
        return relevant





    def _compute_metrics_at_k(self, relevant_links: Set[str],
                             retrieved_links: List[str], k: int) -> Dict[str, float]:
        """
        计算在top-k下的指标

        Args:
            ground_truth_link: ground truth链接
            retrieved_links: 检索到的链接列表
            k: top-k值

        Returns:
            包含hit_rate和mrr的字典
        """
        top_k_links = retrieved_links[:k]
        hits = [1 if link in relevant_links else 0 for link in top_k_links]
        num_hits = sum(hits)

        hit_rate = 1.0 if num_hits > 0 else 0.0

        mrr = 0.0
        for idx, rel in enumerate(hits, start=1):
            if rel:
                mrr = 1.0 / idx
                break

        precision = num_hits / k if k > 0 else 0.0
        recall_denominator = len(relevant_links) if relevant_links else 0
        recall = num_hits / min(recall_denominator, k) if recall_denominator else 0.0
        if recall_denominator and recall_denominator < k:
            # 当相关文档少于k时，recall应该基于真实相关数量
            recall = num_hits / recall_denominator

        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)

        ndcg = self._compute_ndcg(hits, k, len(relevant_links))
        average_precision = self._compute_average_precision(hits, len(relevant_links), k)

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
            "average_precision": average_precision,
            "k": k
        }

    def _compute_ndcg(self, hits: List[int], k: int, num_relevant: int) -> float:
        """计算NDCG@k"""
        dcg = 0.0
        for idx, rel in enumerate(hits):
            if rel:
                dcg += 1.0 / math.log2(idx + 2)

        ideal_length = min(num_relevant, k)
        if ideal_length == 0:
            return 0.0

        idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_length))
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_average_precision(self, hits: List[int], num_relevant: int, k: int) -> float:
        """计算AP@k"""
        if num_relevant == 0:
            return 0.0

        precisions = []
        hit_count = 0
        for idx, rel in enumerate(hits, start=1):
            if rel:
                hit_count += 1
                precisions.append(hit_count / idx)

        if not precisions:
            return 0.0

        denominator = min(num_relevant, k)
        return sum(precisions) / denominator if denominator else 0.0

    def print_evaluation_summary(self, summary: Dict[str, Any]) -> None:
        """打印评估摘要"""
        print("\n" + "="*50)
        print("检索评估结果摘要")
        print("="*50)
        print(f"总查询数: {summary['num_queries']}")
        aggregated = summary.get("aggregated_metrics", {})

        # 统计有效查询数量
        valid_counts = [stats.get("num_valid_queries", 0) for stats in aggregated.values()]
        valid_queries = max(valid_counts) if valid_counts else 0
        print(f"有效查询数: {valid_queries}")

        metrics_by_k: Dict[int, Dict[str, Dict[str, float]]] = {}
        for key, stats in aggregated.items():
            if "@" not in key:
                continue
            metric_name, k_str = key.split("@", maxsplit=1)
            try:
                k = int(k_str)
            except ValueError:
                continue
            metrics_by_k.setdefault(k, {})
            metrics_by_k[k][metric_name] = stats

        print("\n各指标结果:")
        for k in sorted(metrics_by_k.keys()):
            metric_stats = metrics_by_k[k]
            valid = metric_stats.get("hit_rate", {}).get("num_valid_queries", 0)
            print(f"Top-{k}:")
            print(f"  Valid Queries: {valid}")
            for metric_name, stats in sorted(metric_stats.items()):
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 0.0)
                print(f"  {metric_name}: {mean:.3f} +/- {std:.3f}")

        print("\n详细结果已保存在individual_results中")

    def save_results(self, summary: Dict[str, Any], output_file: str) -> None:
        """保存评估结果到文件"""
        import json
        from pathlib import Path

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存到: {output_file}")

    def compare_methods(self, method_results: Dict[str, Dict[str, Any]],
                       top_k: int = 5) -> None:
        """
        比较不同检索方法的性能

        Args:
            method_results: 方法名称到评估结果的映射
            top_k: 比较的top-k值
        """
        print("\n" + "="*60)
        print(f"检索方法性能对比 (Top-{top_k})")
        print("="*60)

        header = f"{'Method':<25}{'Hit Rate':>12}{'MRR':>12}{'Valid':>8}"
        print(header)
        print("-" * len(header))

        metric_key_hit = f"hit_rate@{top_k}"
        metric_key_mrr = f"mrr@{top_k}"

        for method_name, summary in method_results.items():
            aggregated = summary.get('aggregated_metrics', {})
            hit_stats = aggregated.get(metric_key_hit, {})
            mrr_stats = aggregated.get(metric_key_mrr, {})
            if not hit_stats and not mrr_stats:
                continue
            hit_rate = hit_stats.get('mean', 0.0)
            mrr = mrr_stats.get('mean', 0.0)
            num_queries = hit_stats.get('num_valid_queries', 0)
            print(f"{method_name:<25}{hit_rate:>12.3f}{mrr:>12.3f}{num_queries:>8}")

        print("-" * len(header))
