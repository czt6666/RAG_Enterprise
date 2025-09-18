# doc_loader.py
from pathlib import Path
from typing import List
from langchain.docstore.document import Document

# LangChain community loaders (new imports)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)

# optional: charset-normalizer for robust encoding detection
try:
    from charset_normalizer import from_bytes
except Exception:
    from_bytes = None  # we will fallback to trying known encodings

class DocumentLoader:
    """
    Load various file formats into LangChain Document objects.
    Robust handling for text files (encoding autodetect + fallbacks).
    """

    def __init__(self, directory: str = None,
                 text_loader_encoding: str | None = None,
                 autodetect_encoding: bool = True,
                 show_progress: bool = False):
        """
        :param directory: optional base directory for load_directory
        :param text_loader_encoding: prefered encoding to pass to TextLoader (None => system default)
        :param autodetect_encoding: whether to pass autodetect flag to TextLoader
        """
        self.directory = Path(directory) if directory else None
        self.text_loader_encoding = text_loader_encoding
        self.autodetect_encoding = autodetect_encoding
        self.show_progress = show_progress

    def _load_text_with_fallback(self, p: Path) -> List[Document]:
        """
        Try TextLoader with autodetect first; if it fails, fallback to charset detection
        and manual decoding attempts.
        """
        # 1) Try TextLoader (this already supports autodetect_encoding in many langchain versions)
        try:
            loader = TextLoader(str(p), encoding=self.text_loader_encoding, autodetect_encoding=self.autodetect_encoding)
            docs = loader.load()
            # add detected encoding info if available (TextLoader may add it)
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source_file", str(p))
            return docs
        except Exception as e:
            # fallback path (file might be in other encoding or contain non-text bytes)
            raw = p.read_bytes()

            # 2) Use charset-normalizer if available
            detected_enc = None
            if from_bytes is not None:
                try:
                    res = from_bytes(raw)
                    best = res.best()  # CharsetMatch or None
                    if best:
                        detected_enc = best.encoding
                except Exception:
                    detected_enc = None

            # 3) Try detected encoding first, then common encodings
            tried = []
            if detected_enc:
                tried.append(detected_enc)
                try:
                    text = raw.decode(detected_enc, errors="replace")
                    return [Document(page_content=text, metadata={"source_file": str(p), "detected_encoding": detected_enc})]
                except Exception:
                    pass

            # common fallbacks
            for enc in ("utf-8", "utf-16", "gbk", "latin-1"):
                if enc in tried:
                    continue
                try:
                    text = raw.decode(enc, errors="replace")
                    return [Document(page_content=text, metadata={"source_file": str(p), "decoded_with": enc})]
                except Exception:
                    continue

            # last resort: decode with 'utf-8' replace
            text = raw.decode("utf-8", errors="replace")
            return [Document(page_content=text, metadata={"source_file": str(p), "decoded_with": "utf-8-replace"})]

    def load_file(self, file_path: str) -> List[Document]:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"{p} does not exist or is not a file")

        suffix = p.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
            docs = loader.load()
        elif suffix == ".docx":
            # try unstructured loader first (if unstructured package present), else docx2txt
            try:
                loader = UnstructuredWordDocumentLoader(str(p))
                docs = loader.load()
            except Exception:
                loader = Docx2txtLoader(str(p))
                docs = loader.load()
        elif suffix == ".txt":
            docs = self._load_text_with_fallback(p)
        elif suffix == ".csv":
            loader = CSVLoader(str(p))
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # ensure metadata has source_file
        for doc in docs:
            if not getattr(doc, "metadata", None):
                doc.metadata = {}
            doc.metadata.setdefault("source_file", str(p))

        return docs

    def load_directory(self, glob_pattern: str = "**/*") -> List[Document]:
        if not self.directory:
            raise ValueError("directory not set for load_directory")
        all_docs = []
        for p in sorted(self.directory.glob(glob_pattern)):
            if p.is_file():
                try:
                    docs = self.load_file(str(p))
                    all_docs.extend(docs)
                except Exception as e:
                    # skip unsupported/broken files but log
                    print(f"[skip] {p}: {e}")
        return all_docs


# quick test runner
if __name__ == "__main__":
    loader = DocumentLoader()
    # replace with your test paths
    for fp in ["example.txt", "example.pdf", "example.docx"]:
        try:
            docs = loader.load_file(fp)
            print(f"Loaded {len(docs)} docs from {fp}; metadata example: {docs[0].metadata if docs else None}")
        except Exception as e:
            print(f"Error loading {fp}: {e}")
