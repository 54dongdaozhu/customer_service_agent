"""测试完整的 RAG 链"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils.model import get_llm
from src.rag.retriever import ProductRetriever


def build_rag_chain():
    """构建完整的 RAG 链"""
    llm = get_llm(temperature=0.3)
    retriever = ProductRetriever(top_k=3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的客服助手。请根据提供的参考资料回答用户问题。

要求：
1. 回答要准确，严格基于参考资料
2. 如果参考资料中没有相关信息，就直接说"抱歉，我没有找到相关信息"
3. 回答要简洁清晰，有条理
4. 适当使用 emoji 让回答更友好

参考资料：
{context}
"""),
        ("human", "{question}"),
    ])

    # 构建链
    rag_chain = (
        {
            "context": lambda x: retriever.retrieve_as_context(x["question"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    rag_chain = build_rag_chain()

    questions = [
        "iPhone 15 有几种颜色？分别是什么？",
        "MacBook Pro 16 英寸多少钱？",
        "iPhone 15 怎么退货？",
        "MacBook 的保修期是多久？",
        "你们卖蔬菜吗？",  # 故意问一个无关问题
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {q}")
        print('='*60)
        answer = rag_chain.invoke({"question": q})
        print(f"💬 回答: {answer}")


if __name__ == "__main__":
    main()
