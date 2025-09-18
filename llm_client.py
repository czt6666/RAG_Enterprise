from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class LLMClient:
    """
    LLM 调用模块
    固定模板 Prompt
    """

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.0):
        # self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt_template = """请根据以下资料回答问题。只使用提供的资料，不要编造。
资料：
{context}
问题：
{question}"""

    def generate_answer(self, question: str, retrieved_docs: list):
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = self.prompt_template.format(context=context, question=question)
        return prompt
        # return self.llm.predict(prompt)


# 测试接口
if __name__ == "__main__":
    from doc_loader import DocumentLoader
    from text_splitter import TextChunker
    from embedder import Embedder
    from chroma_store import ChromaStore
    from retriever import Retriever

    loader = DocumentLoader(directory="files")
    docs = loader.load_directory()
    chunks = TextChunker().chunk_documents(docs)
    embedder = Embedder()
    store = ChromaStore()
    store.build_store(chunks, embedder=embedder)

    retriever = Retriever(store)
    query = "常志韬的工作经历是什么"
    llm = LLMClient()
    top_docs = retriever.retrieve(query)
    answer = llm.generate_answer(query, top_docs)
    print(answer)
