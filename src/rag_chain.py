import pickle
import jieba
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import Config

def jieba_preprocess(text: str) -> list[str]:
    return jieba.lcut(text)

def get_rag_chain(use_rerank: bool = True):
    """
    构建 RAG 链

    Args:
        use_rerank: 是否使用 Rerank，默认为 True

    Returns:
        RAG 链
    """
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        Config.VECTOR_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    with open(Config.DOCS_DIR, "rb") as f:
        docs = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocess)
    bm25_retriever.k = 10

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    # LLM
    llm = ChatOpenAI(
        api_key=Config.get_api_key(),
        base_url=Config.DASHSCOPE_BASE_URL,
        model=Config.llm_model,
        temperature=Config.llm_temperature
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )

    # 根据参数决定是否使用 Rerank
    if use_rerank:
        cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=Config.top_n)

        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever
        )
    else:
        final_retriever = multi_query_retriever

    template = """
    请根据检索到的背景信息回答用户问题。如果背景信息中无法得出答案，请回答"不知道"。
    背景信息：{context}
    用户问题：{question}
    回答：
    """
    prompt = ChatPromptTemplate.from_template(template=template)

    def format_docs(docs):
        """格式化文档并且打印调试信息"""
        for i, d in enumerate(docs):
            print(f"Doc{i+1}: {d.page_content}")
        print("---------------------\n")
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
