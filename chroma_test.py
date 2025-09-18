# inspect_chroma_texts.py

from langchain_community.vectorstores import Chroma
# 或者 from langchain_chroma import Chroma 如果你用那个包

from embedder import Embedder  # 用来初始化 embedding，部分版本需要 embedding 参数加载

def get_all_docs_texts(persist_directory: str = "./chroma_db", collection_name: str = "rag_collection"):
    db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # 使用 get() 方法拉所有文档
    data = db.get(
        include=["documents", "metadatas"],  # 不要 include embeddings
        limit=None  # 或者你知道总条目多少，用一个大数字
    )

    docs = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    ids = data.get("ids", [])

    # 输出每条文档前10个字符
    for i, doc_text in enumerate(docs):
        preview = doc_text[:10].replace("\n", " ")  # 去掉换行影响
        md = metadatas[i] if i < len(metadatas) else {}
        id_ = ids[i] if i < len(ids) else None
        print(f"ID: {id_}, Source: {md.get('source_file','unknown')}, Preview: \"{preview}\"")

if __name__ == "__main__":
    get_all_docs_texts(persist_directory="./chroma_db", collection_name="rag_collection")
