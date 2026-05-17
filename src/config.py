import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_DIR = os.path.join(BASE_DIR, 'vectorstore', 'faiss_index')
    DOCS_DIR = os.path.join(BASE_DIR, 'vectorstore', 'docs.pkl')
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # RAG 配置
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANK_MODEL:str = "BAAI/bge-reranker-base"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    retriever_k: int = 5
    top_n: int = 2

    # LLM 配置
    llm_model: str= "qwen3.6-plus"
    llm_eval: str = "qwen-flash"
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0 # 生成时也应该温度=0
    llm_max_tokens: int = 2000

    # 实验配置
    EXPERIMENT_RETRIEVER_K: int = 10  # 检索候选数
    EXPERIMENT_TOP_N: int = 2  # Rerank 后保留数
    EXPERIMENT_MULTIQUERY_N: int = 3  # MultiQuery 变体数
    EVAL_LLM_MODEL: str = "glm-4-flash"  # RAGAS 评判模型
    EVAL_LLM_TEMPERATURE: float = 0.0  # 评判时温度=0 保证一致性
    @staticmethod
    def get_api_key():
        return os.getenv("DASHSCOPE_API_KEY")