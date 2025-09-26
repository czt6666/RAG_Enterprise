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
    from rag_core.doc_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_file("example.pdf")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

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

import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletionMessage


class LLMClient:
    """
    LLM 调用模块
    固定模板 Prompt
    """

    def __init__(self, api_key=None):
        self.client = OpenAI(
            # api_key=api_key or os.getenv("OPENAI_API_KEY"),
            api_key="sk-cf66836784b34b00b0c6d4034f99fab5",
            base_url="https://api.deepseek.com"
        )
        self.prompt_template = """请根据以下资料回答问题。只使用提供的资料，不要编造。
资料：
{context}
问题：
{question}"""

    def generate_answer(self, question: str, retrieved_docs: list):
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = self.prompt_template.format(context=context, question=question)
        messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        return response.choices[0].message.content


# 测试接口
if __name__ == "__main__":
    # from rag_core.doc_loader import DocumentLoader
    # from rag_core.text_splitter import TextChunker
    # from rag_core.embedder import Embedder
    # from rag_core.chroma_store import ChromaStore
    # from rag_core.retriever import Retriever
    from types import SimpleNamespace
    #
    # loader = DocumentLoader(directory="files")
    # docs = loader.load_directory()
    # chunks = TextChunker().chunk_documents(docs)
    # embedder = Embedder()
    # store = ChromaStore(persist_directory="../chroma_db")
    # store.build_store(chunks, embedder=embedder)
    #
    # retriever = Retriever(store)
    query = "常志韬的工作经历是什么"
    llm = LLMClient()
    # top_docs = retriever.retrieve(query)
    top_docs = [
        {
        "page_content": "常志韬建立前端代码规范，使用ESLint和Prettier实现自动化代码格式校验。将项目从Webpack迁移至Vite，实现秒级启动，打包时间从19秒优化至9秒。使用Node.js编写构建工具，将多文件代码打包为单HTML文件，服务于大模型训练数据。开发HTML自动化测试工具，用于检测转化后的文件中JavaScript报错与资源请求失败，确保HTML文件可运行。24岁｜男｜一年经验13521670204｜changzhitao12@126.com教育经历北京信息科技大学&北京理工大学联培-物联网工程-本科2020-09~2024-07技术栈精通HTML、CSS、JavaScript、Node.js、TypeScript、SCSS等Web开发技术。",
    }
    ]

    top_docs = [SimpleNamespace(**d) if isinstance(d, dict) else d for d in top_docs]

    answer = llm.generate_answer(query, top_docs)
    print(answer)
