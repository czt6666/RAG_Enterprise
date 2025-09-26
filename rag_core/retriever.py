from rag_core.chroma_store import ChromaStore

class Retriever:
    """
    检索模块
    Top-K 查询
    """
    def __init__(self, chroma_store: ChromaStore, top_k=3):
        self.store = chroma_store
        self.top_k = top_k

    def retrieve(self, query_text: str):
        return self.store.query(query_text, k=self.top_k)

# 测试接口
if __name__ == "__main__":
    store = ChromaStore(persist_directory="../chroma_db")
    retriever = Retriever(store)
    docs = retriever.retrieve("测试问题")
    for d in docs:
        print(d.page_content[:100])
