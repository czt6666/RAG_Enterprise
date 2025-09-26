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
