import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path


class DataLoader:
    """数据加载器，用于加载查询和数据集文件"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.queries = []
        self.datasets = []
        self._loaded = False

    def load_data(self) -> None:
        """加载所有数据"""
        self._load_queries()
        self._load_datasets()
        self._loaded = True
        print(f"Loaded {len(self.queries)} queries and {len(self.datasets)} datasets")

    def _load_queries(self) -> None:
        """加载查询数据"""
        query_file = self.data_dir / "query.json"
        if not query_file.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")

        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                self.queries = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load queries: {e}")

    def _load_datasets(self) -> None:
        """加载数据集数据"""
        dataset_file = self.data_dir / "final_dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                self.datasets = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load datasets: {e}")

    def get_queries(self) -> List[Dict[str, Any]]:
        """获取所有查询"""
        if not self._loaded:
            self.load_data()
        return self.queries

    def get_datasets(self) -> List[Dict[str, Any]]:
        """获取所有数据集"""
        if not self._loaded:
            self.load_data()
        return self.datasets

    def get_query_by_index(self, index: int) -> Dict[str, Any]:
        """根据索引获取查询"""
        if not self._loaded:
            self.load_data()
        if 0 <= index < len(self.queries):
            return self.queries[index]
        else:
            raise IndexError(f"Query index {index} out of range")

    def get_dataset_by_index(self, index: int) -> Dict[str, Any]:
        """根据索引获取数据集"""
        if not self._loaded:
            self.load_data()
        if 0 <= index < len(self.datasets):
            return self.datasets[index]
        else:
            raise IndexError(f"Dataset index {index} out of range")

    def get_ground_truth_link(self, query: Dict[str, Any]) -> str:
        """从查询中提取ground truth链接"""
        if "ground_truth" in query and "link" in query["ground_truth"]:
            return query["ground_truth"]["link"]
        return ""

    def get_ground_truth_name(self, query: Dict[str, Any]) -> str:
        """从查询中提取ground truth数据集名称"""
        if "ground_truth" in query and "name" in query["ground_truth"]:
            return query["ground_truth"]["name"]
        return ""

    def find_dataset_by_link(self, link: str) -> Dict[str, Any]:
        """根据链接查找对应的数据集"""
        if not self._loaded:
            self.load_data()

        for dataset in self.datasets:
            if dataset.get("Dataset Link") == link:
                return dataset
        return {}

    def get_positive_links(self, query: Dict[str, Any]) -> List[str]:
        """获取查询的正例链接"""
        return query.get("positives", [])

    def get_instruction(self, query: Dict[str, Any]) -> str:
        """获取查询的指令"""
        return query.get("instruction", "")

    def get_query_text(self, query: Dict[str, Any]) -> str:
        """获取查询文本"""
        return query.get("query", "")

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self._loaded:
            self.load_data()

        stats = {
            "num_queries": len(self.queries),
            "num_datasets": len(self.datasets),
            "query_fields": set(),
            "dataset_fields": set()
        }

        if self.queries:
            stats["query_fields"] = set(self.queries[0].keys())

        if self.datasets:
            stats["dataset_fields"] = set(self.datasets[0].keys())

        return stats

    def __len__(self) -> int:
        """返回查询数量"""
        if not self._loaded:
            self.load_data()
        return len(self.queries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """通过索引访问查询"""
        return self.get_query_by_index(index)
