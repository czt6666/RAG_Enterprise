# chroma_test.py
from rag_core.embedder import Embedder
from rag_core.chroma_store import ChromaStore
import os
import re

class ChromaTest:
    def __init__(self, persist_directory="../chroma_db", collection_name="rag_collection"):
        self.embedder = Embedder()
        self.store = ChromaStore(persist_directory=persist_directory, collection_name=collection_name)
        self.store.load_store(embedder=self.embedder)

    def show_all(self, n=None, limit=None):
        """
        打印数据库中的全部内容
        格式：【文件名】内容前n个字
        :param n: 单条内容显示的最大字符数（None=不限制）
        :param limit: 最大展示数量（None=展示所有）
        """
        all_data = self.store.db.get(include=["documents", "metadatas"])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            if limit is not None and i >= limit:
                print(f"... 已省略 {len(docs) - limit} 条")
                break

            filename = os.path.basename(meta.get("source_file", "未知文件"))
            clean_text = re.sub(r"\s+", "", doc)
            preview = clean_text if n is None else clean_text[:n]
            print(f"{i+1}[{filename}]{preview}")

# 测试接口
if __name__ == "__main__":
    tester = ChromaTest()
    tester.show_all(n=50)
