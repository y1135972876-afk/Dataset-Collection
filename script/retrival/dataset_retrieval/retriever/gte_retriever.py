import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
from .base_retriever import BaseRetriever


class GTERetriever(BaseRetriever):
    """基于Alibaba-NLP GTE模型的检索器"""

    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5", device: str = None):
        super().__init__(model_name)

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def initialize(self) -> None:
        """初始化GTE模型和tokenizer"""
        try:
            print(f"Loading GTE model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

            # 将模型移动到指定设备
            self.model.to(self.device)
            self.model.eval()

            self.is_initialized = True
            print(f"GTE model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GTE model: {e}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """批量编码文本为向量"""
        if not self.is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        embeddings = []

        # 分批处理，避免内存溢出
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            embeddings.append(batch_embeddings)

        if embeddings:
            return np.concatenate(embeddings, axis=0)
        else:
            return np.array([])

    def encode_text(self, text: str) -> np.ndarray:
        """编码单个文本为向量"""
        if not self.is_initialized:
            raise RuntimeError("Retriever must be initialized before use")

        return self._encode_batch([text])[0]

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """内部方法：编码一批文本"""
        # 准备输入
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")

        # 将输入移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # 使用[CLS] token的表示

        # 归一化向量
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 返回numpy数组
        return embeddings.cpu().numpy()

    def _extract_search_text(self, candidate: Dict[str, Any]) -> str:
        """将候选文档的全部字段拼接为检索文本"""
        def format_value(value: Any) -> str:
            if isinstance(value, dict):
                parts = []
                for key in sorted(value.keys()):
                    nested = format_value(value[key])
                    if nested:
                        parts.append(f"{key}: {nested}")
                return " ".join(parts)
            if isinstance(value, (list, tuple, set)):
                return " ".join(format_value(item) for item in value)
            if value is None:
                return ""
            return str(value)

        sections = []
        for key in sorted(candidate.keys()):
            formatted = format_value(candidate[key])
            if formatted:
                sections.append(f"{key}: {formatted}")

        return " ".join(sections)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "model_type": "GTE",
            "max_sequence_length": 512
        }
