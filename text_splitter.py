from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

class TextChunker:
    """
    文本切分模块
    固定长度 + 滑动窗口切分
    """
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        chunks = []
        for doc in docs:
            split_texts = self.splitter.split_text(doc.page_content)
            for t in split_texts:
                chunks.append(Document(page_content=t, metadata=doc.metadata))
        return chunks

# 测试接口
if __name__ == "__main__":
    from doc_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_file("example.pdf")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")
