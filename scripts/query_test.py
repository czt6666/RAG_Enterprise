# query_test.py

from rag_core.chroma_store import ChromaStore
from rag_core.embedder import Embedder
from rag_core.retriever import Retriever
from rag_core.llm_client import LLMClient
import os
import re


DB_DIR = "../chroma_db"
COLLECTION = "rag_collection"

def main():
    query = "常志韬的工作经历是什么"

    # 1. 加载向量库
    embedder = Embedder()
    store = ChromaStore(persist_directory=DB_DIR, collection_name=COLLECTION)
    store.load_store(embedder=embedder)

    # 2. 用 retriever 做检索
    retriever = Retriever(store)
    results = retriever.retrieve(query)

    print("=== 用户问题 ===")
    print(query)
    print("\n=== 检索到的文档 ===")
    for i, d in enumerate(results, 1):
        filename = os.path.basename(d.metadata.get("source_file", "未知文件"))
        file_text = re.sub(r"\s+", "", d.page_content)
        print(f"[{i}]{filename}|{file_text}")

    # 3. 送到 LLM
    llm = LLMClient()
    answer = llm.generate_answer(query, results)
    print("\n=== LLM 回答 ===")
    print(answer)

if __name__ == "__main__":
    main()
