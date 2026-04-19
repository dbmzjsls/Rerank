import os.path
from src.bulid_db import build_and_save_db
from src.rag_chain import get_rag_chain


def main():
    from src.config import Config
    if not os.path.exists(Config.VECTOR_DIR):
        print("正在构建向量数据库...")
        build_and_save_db()
        print("\n")

    print("发现已存在向量数据库\n")
    print("正在初始化 RAG 链路（加载模型会需要一些时间）...")
    chain = get_rag_chain()
    print("初始化完成！\n")

    while True:
        question = input("请提问（输入'q'即可退出）:")
        if question.lower() == "q":
            print("小助手下班了，欢迎下次光临\n━(*｀∀´*)ノ亻!")
            break

        print("小助手正在思考中...")
        response = chain.invoke(question)
        print(f"回答:{response}\n")

if __name__ == "__main__":
    main()