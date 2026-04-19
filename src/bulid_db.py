import os
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

def build_and_save_db():
    loader = PyPDFDirectoryLoader(Config.DATA_DIR)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    docs = splitter.split_documents(raw_docs)

    print(f"共加载 {len(raw_docs)} 页，分割为 {len(docs)} 个文本块")

    print("正在初始化 Embeddings 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

    print("正在构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(os.path.dirname(Config.VECTOR_DIR), exist_ok=True)
    vectorstore.save_local(Config.VECTOR_DIR)

    with open(Config.DOCS_DIR, 'wb') as f:
        pickle.dump(docs, f)

    print("数据库构建完成!")

if __name__ == "__main__":
    build_and_save_db()
