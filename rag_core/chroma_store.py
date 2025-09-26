# chroma_store.py

from pathlib import Path
from typing import List, Optional
from langchain.docstore.document import Document

# 新版导入
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

class ChromaStore:
    """
    向量数据库模块
    """

    def __init__(self, persist_directory="./chroma_db", collection_name="rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.db = None  # 初始化为空，等 build_store 再赋值

    def build_store(self, docs: List[Document], embedder):
        """
        docs: List[Document]
        embedder: 一个实现了 embed_documents / embed_query 的对象（比如 Embedder）
        """
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=embedder,   # ⚠️ 注意，这里传的是对象，不是函数
            metadatas=metadatas,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

    def load_store(self, embedder):
        self.db = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=embedder
        )

    def query(self, query_text: str, k=3):
        return self.db.similarity_search(query_text, k=k)


# 测试接口
if __name__ == "__main__":
    from rag_core.doc_loader import DocumentLoader
    from rag_core.text_splitter import TextChunker
    from rag_core.embedder import Embedder

    loader = DocumentLoader()
    docs = loader.load_file("example.pdf")
    chunks = TextChunker().chunk_documents(docs)
    embedder = Embedder()
    # vectors = embedder.embed_documents([c.page_content for c in chunks])

    store = ChromaStore(persist_directory="../chroma_db")
    store.build_store(chunks, embedder)
    res_docs = store.query("测试问题", k=3)

    for d in res_docs:
        print("------")
        print(d.page_content[:200])
        print(d.metadata)
