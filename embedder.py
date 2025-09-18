# embedder.py

from typing import List, Optional
import os
import torch
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class Embedder(Embeddings):
    """
    向量化模块
    使用 BGE 中文模型（或其他模型）做本地 embedding，实现 embed_documents 和 embed_query，
    兼容 Chroma.from_texts / from_documents 的接口要求。
    """

    def __init__(self,
                 model_name: str = "BAAI/bge-base-zh-v1.5",
                 device: Optional[str] = None):
        """
        :param model_name: 主 embedding 模型名
        :param device: "cuda", "cpu" 等。如果为 None，会自动选 cuda（如果可用）或 cpu
        """
        super().__init__()  # 初始化父类（虽然父类可能没什么做，但形式上做一下）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        try:
            print(f"[Embedder] Loading model {self.model_name} on device {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"[Embedder] Failed to load model {self.model_name}: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        给多个文档文本做 embedding
        返回 List[List[float]]
        """
        # 去除换行符之类的（可选）
        cleaned = [t.replace("\n", " ") if isinstance(t, str) else "" for t in texts]
        embeddings = self.model.encode(cleaned, show_progress_bar=True, convert_to_tensor=False)  # 返回 numpy array 或 list
        # 确保是 list[list[float]]
        # sentence_transformers.encode 返回 numpy 或 list，直接 list 就好
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        给单条查询做 embedding
        返回 List[float]
        """
        # 单元素列表操作
        emb = self.model.encode(text, show_progress_bar=False, convert_to_tensor=False)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb


# 测试接口
if __name__ == "__main__":
    embedder = Embedder()
    texts = ["你好，世界", "RAG 系统测试"]
    vecs = embedder.embed_documents(texts)
    print("emb docs shapes:", len(vecs), len(vecs[0]))
    q = "测试问题"
    qv = embedder.embed_query(q)
    print("emb query length:", len(qv))
