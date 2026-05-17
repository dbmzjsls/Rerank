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


def _load_vectorstore_and_docs():
    """加载向量库和文档"""
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        Config.VECTOR_DIR,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    with open(Config.DOCS_DIR, "rb") as f:
        docs = pickle.load(f)
    return vectorstore, docs


def _build_retriever(vectorstore, docs, mode: str):
    """构建检索器"""
    # 纯向量检索
    if mode == "vector":
        return vectorstore.as_retriever(search_kwargs={"k": Config.retriever_k})

    # 混合检索 (BM25 + FAISS)
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=jieba_preprocess)
    bm25.k = Config.EXPERIMENT_RETRIEVER_K
    vec_ret = vectorstore.as_retriever(search_kwargs={"k": Config.EXPERIMENT_RETRIEVER_K})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vec_ret],
        weights=[0.4, 0.6],
    )

    if mode == "ensemble":
        return ensemble

    if mode == "rerank":
        cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RERANK_MODEL)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=Config.top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )

    if mode == "multiquery":
        llm_for_query = ChatOpenAI(
            api_key=Config.get_api_key(),
            base_url=Config.DASHSCOPE_BASE_URL,
            model=Config.llm_model,
            temperature=0.3,
        )
        return MultiQueryRetriever.from_llm(
            retriever=ensemble,
            llm=llm_for_query,
        )

    raise ValueError(f"不支持的检索模式: {mode}，有效选项: vector, ensemble, rerank, multiquery")


def get_rag_chain(mode: str = "ensemble"):
    """
    构建 RAG 链，返回 LCEL Runnable。
    """
    vectorstore, docs = _load_vectorstore_and_docs()
    retriever = _build_retriever(vectorstore, docs, mode)

    llm = ChatOpenAI(
        api_key=Config.get_api_key(),
        base_url=Config.DASHSCOPE_BASE_URL,
        model=Config.llm_model,
        temperature=Config.llm_temperature,
    )

    template = (
        "请根据以下背景信息回答用户问题。"
        "如果背景信息中无法得出答案，请明确说明。\n\n"
        "背景信息：\n{context}\n\n"
        "用户问题：{question}\n\n回答："
    )
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        for i, d in enumerate(docs, 1):
            preview = d.page_content[:80].replace("\n", " ")
            print(f"  Doc{i}: {preview}...")
        print("---------------------\n")
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ── 供 RAGAS 评估使用的独立组件 ──────────────────────

def get_retriever_only(mode: str = "ensemble"):
    """
    仅获取检索器对象，仅在参数评估中使用
    用法: retriever = get_retriever_only("rerank"); docs = retriever.invoke(question)
    """
    vectorstore, docs = _load_vectorstore_and_docs()
    return _build_retriever(vectorstore, docs, mode)


def get_llm_only():
    """
    仅获取基础 LLM 实例,仅在参数评估中使用
    用法: llm = get_llm_only(); answer = llm.invoke(prompt).content
    """
    return ChatOpenAI(
        api_key=Config.get_api_key(),
        base_url=Config.DASHSCOPE_BASE_URL,
        model=Config.llm_model,
        temperature=0,
    )
