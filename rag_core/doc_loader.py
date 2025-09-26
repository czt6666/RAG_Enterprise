# doc_loader.py
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from charset_normalizer import from_bytes

# LangChain community loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)

class DocumentLoader:
    """
    加载目录下的多种文件，返回统一的 List[Document]
    约定大于配置：只要传 directory，就能直接 load_directory()
    """

    def __init__(self, directory: str):
        self.directory = Path(directory)
        if not self.directory.exists() or not self.directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

    def _load_text_with_fallback(self, p: Path) -> List[Document]:
        try:
            loader = TextLoader(str(p), autodetect_encoding=True)
            docs = loader.load()
        except Exception:
            raw = p.read_bytes()
            # charset_normalizer 检测编码
            detected_enc = None
            try:
                res = from_bytes(raw)
                best = res.best()
                if best:
                    detected_enc = best.encoding
            except Exception:
                pass
            if detected_enc:
                text = raw.decode(detected_enc, errors="replace")
            else:
                text = raw.decode("utf-8", errors="replace")
            docs = [Document(page_content=text)]
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source_file", str(p))
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"{p} does not exist or is not a file")

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            docs = PyPDFLoader(str(p)).load()
        elif suffix == ".docx":
            try:
                docs = UnstructuredWordDocumentLoader(str(p)).load()
            except Exception:
                docs = Docx2txtLoader(str(p)).load()
        elif suffix == ".txt":
            docs = self._load_text_with_fallback(p)
        elif suffix == ".csv":
            docs = CSVLoader(str(p)).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source_file", str(p))
        return docs

    def load_directory(self, glob_pattern: str = "**/*") -> List[Document]:
        all_docs = []
        for p in sorted(self.directory.glob(glob_pattern)):
            if p.is_file():
                try:
                    all_docs.extend(self.load_file(str(p)))
                except Exception as e:
                    print(f"[skip] {p}: {e}")
        return all_docs


if __name__ == "__main__":
    # 示例：直接传目录，一步加载全部
    loader = DocumentLoader("../files")
    docs = loader.load_directory()
    print(f"共加载 {len(docs)} 个 Document")
    if docs:
        print("示例内容:", docs[0].page_content[:50])
        print("示例元数据:", docs[0].metadata)
