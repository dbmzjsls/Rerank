# Rerank RAG 项目

基于 LangChain 的检索增强生成（RAG）系统，集成混合检索 + 查询重写 + 重排序（Rerank）的漏斗式检索流水线。

## 架构

```
用户提问
  └─> MultiQueryRetriever（查询重写，生成多条 query）
        └─> EnsembleRetriever（混合检索）
              ├─> BM25Retriever（稀疏检索，字面匹配）
              └─> FAISS VectorRetriever（稠密检索，语义匹配）
        └─> CrossEncoderReranker（重排序，精选 top_n 文档）
  └─> LLM 生成回答
```

## 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器
- DashScope API Key（阿里云百炼）

## 快速开始

**1. 安装依赖**
```bash
uv sync
```

**2. 配置环境变量**

创建 `.env` 文件：
```
DASHSCOPE_API_KEY=your_api_key_here
```

**3. 添加文档**

将 PDF 文件放入 `data/` 目录：
```
data/
├── 文档1.pdf
└── 文档2.pdf
```

**4. 运行**
```bash
uv run python main.py
```

首次运行会自动构建向量数据库，之后直接加载。

## 项目结构

```
Rerank/
├── data/               # 放置 PDF 文档
├── vectorstore/        # 自动生成，存储向量库和文档
│   ├── faiss_index/
│   └── docs.pkl
├── src/
│   ├── config.py       # 全局配置
│   ├── bulid_db.py     # 构建向量数据库
│   └── rag_chain.py    # RAG 检索链
├── main.py
└── pyproject.toml
```

## 配置说明

`src/config.py` 中可调整的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 文本分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠字符数 |
| `retriever_k` | 5 | 检索返回文档数 |
| `top_n` | 2 | Rerank 后保留文档数 |
| `llm_model` | qwen-plus | LLM 模型 |
| `llm_temperature` | 0.2 | 生成温度 |

## 重建向量数据库

更换文档后需删除旧数据库：
```bash
rm -rf vectorstore/
uv run python main.py
```
