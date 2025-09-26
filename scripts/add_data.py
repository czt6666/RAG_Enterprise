# add_data.py
from rag_core.doc_loader import DocumentLoader
from rag_core.text_splitter import TextChunker
from rag_core.embedder import Embedder
from rag_core.chroma_store import ChromaStore

# 配置
DATA_DIR = "../files"          # 你的文档目录
DB_DIR = "../chroma_db"       # 向量数据库保存位置
COLLECTION = "rag_collection"

def main():
    # 1. 加载文件
    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_directory()
    print(f"共加载 {len(docs)} 个文档 chunk")

    # 2. 切分
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"切分后得到 {len(chunks)} 个 chunk")

    # 3. 向量化 + 写入数据库
    embedder = Embedder()
    store = ChromaStore(persist_directory=DB_DIR, collection_name=COLLECTION)
    store.build_store(chunks, embedder)
    print(f"已写入数据库 {DB_DIR}/{COLLECTION}")

if __name__ == "__main__":
    main()
