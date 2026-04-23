"""产品相关工具（封装 RAG 检索为工具）"""
from langchain_core.tools import tool
# 依赖包：langchain_core

from src.rag.retriever import ProductRetriever


# 单例检索器
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = ProductRetriever(top_k=3)
    return _retriever


@tool
def search_product_info(query: str) -> str:
    """
    从产品知识库中搜索相关信息。适用于回答以下类型的问题：
    - 产品参数、功能、规格
    - 产品价格
    - 使用方法、操作指引
    - 退换货政策、保修政策
    - 产品型号对比

    Args:
        query: 用户的查询内容，建议是具体的问题或关键词

    Returns:
        从产品知识库中检索到的相关内容
    """
    retriever = _get_retriever()
    context = retriever.retrieve_as_context(query)
    return context


if __name__ == "__main__":
    queries = [
        "iPhone 15 有哪些颜色？",
        "MacBook Pro 的价格",
        "退货流程是什么",
    ]

    for q in queries:
        print(f"\n{'='*50}")
        print(f"🔍 查询: {q}")
        print('='*50)
        result = search_product_info.invoke({"query": q})
        print(result)
