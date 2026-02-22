from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class BaseRetriever(ABC):
    """检索器基类，定义检索器的基本接口"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_initialized = False
        self._candidate_embeddings: Optional[np.ndarray] = None
        self._candidate_cache_size: int = 0
        self._use_faiss: bool = False
        self._faiss_index = None
        self._faiss_module = None
        self._faiss_warning_emitted = False

    @abstractmethod
    def initialize(self) -> None:
        """初始化检索模型"""
        pass

    @abstractmethod
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """将单个文本编码为向量"""
        pass

    def configure_indexing(self, use_faiss: bool = False) -> None:
        """配置是否使用Faiss索引"""
        if use_faiss:
            if self._faiss_module is None:
                try:
                    import faiss  # type: ignore
                    self._faiss_module = faiss
                except ImportError:
                    if not self._faiss_warning_emitted:
                        print("Faiss 未安装，使用numpy检索作为回退。")
                        self._faiss_warning_emitted = True
                    self._faiss_module = False  # type: ignore[assignment]
                    use_faiss = False
            elif self._faiss_module is False:
                use_faiss = False
        else:
            self._faiss_index = None

        self._use_faiss = use_faiss

    def is_faiss_enabled(self) -> bool:
        """判断当前是否启用了Faiss检索"""
        return self._use_faiss and self._faiss_module not in (None, False)

    def prepare_candidates(self, candidates: List[Dict[str, Any]], rebuild: bool = False) -> None:
        """预编码候选文档，并在需要时构建索引"""
        if not self.is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        needs_encoding = (
            rebuild
            or self._candidate_embeddings is None
            or len(candidates) != self._candidate_cache_size
        )

        if needs_encoding:
            candidate_texts = [self._extract_search_text(candidate) for candidate in candidates]
            if candidate_texts:
                print(f"Encoding {len(candidate_texts)} candidates...")
                embeddings = self.encode_texts(candidate_texts).astype(np.float32)
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)

            self._candidate_embeddings = embeddings
            self._candidate_cache_size = len(candidates)
            self._faiss_index = None  # 索引需要重新构建

        if (
            self._use_faiss
            and self._faiss_module not in (None, False)
            and self._faiss_index is None
            and self._candidate_embeddings is not None
            and self._candidate_embeddings.size > 0
        ):
            print("Building Faiss index for candidates...")
            self._build_faiss_index()

    def retrieve(self, query: str, candidates: List[Dict[str, Any]],
                 instruction: str = None, top_k: int = 10,
                 with_instruction: bool = False,
                 use_faiss: Optional[bool] = None,
                 rebuild_index: bool = False) -> List[Tuple[Dict[str, Any], float]]:
        """
        检索最相关的候选文档

        Args:
            query: 查询文本
            candidates: 候选文档列表，每个文档包含文本信息
            instruction: 指令文本（可选）
            top_k: 返回前k个结果
            with_instruction: 是否使用instruction
            use_faiss: 是否启用Faiss索引（默认沿用当前配置）
            rebuild_index: 是否强制重建候选索引

        Returns:
            返回 (文档, 相似度分数) 的列表，按相似度降序排列
        """
        if not self.is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        if use_faiss is not None:
            self.configure_indexing(use_faiss=use_faiss)

        # 构造检索查询
        if with_instruction and instruction:
            retrieval_query = f"{instruction}\n\n{query}"
        else:
            retrieval_query = query

        # 编码查询
        query_embedding = self.encode_text(retrieval_query)

        if not candidates:
            return []

        # 预处理候选文档
        self.prepare_candidates(candidates, rebuild=rebuild_index)

        if self._use_faiss and self._faiss_index is not None:
            query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
            search_k = min(top_k, self._candidate_cache_size)
            scores, indices = self._faiss_index.search(query_vector, search_k)
            results: List[Tuple[Dict[str, Any], float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(candidates):
                    continue
                results.append((candidates[int(idx)], float(score)))
            return results

        # 计算相似度
        if self._candidate_embeddings is None or self._candidate_embeddings.size == 0:
            return []

        similarities = self._compute_similarities(query_embedding, self._candidate_embeddings)

        # 排序并返回top_k结果
        results = []
        for idx, similarity in enumerate(similarities):
            results.append((candidates[idx], float(similarity)))

        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _compute_similarities(self, query_embedding: np.ndarray,
                             candidate_embeddings: np.ndarray) -> np.ndarray:
        """计算查询向量与候选向量之间的相似度"""
        # 使用余弦相似度
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)

        # 避免除零错误
        query_norm = query_norm if query_norm > 0 else 1.0
        candidate_norms = np.where(candidate_norms == 0, 1.0, candidate_norms)

        # 计算余弦相似度
        similarities = np.dot(candidate_embeddings, query_embedding) / (candidate_norms * query_norm)
        return similarities

    @abstractmethod
    def _extract_search_text(self, candidate: Dict[str, Any]) -> str:
        """从候选文档中提取用于检索的文本"""
        pass

    def _build_faiss_index(self) -> None:
        """根据候选向量构建Faiss索引"""
        if self._faiss_module is None or self._candidate_embeddings is None:
            return

        embeddings = self._candidate_embeddings
        if embeddings.size == 0:
            return

        dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)
        index = self._faiss_module.IndexFlatIP(dim)
        index.add(embeddings)
        self._faiss_index = index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
