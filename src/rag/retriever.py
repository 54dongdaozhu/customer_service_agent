"""检索器（封装检索逻辑）"""
from typing import List, Dict

from src.rag.vector_store import search


MAX_CHARS_PER_DOC = 300  # 每段最多保留字符数


class ProductRetriever:
    """产品知识库检索器"""

    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        """检索相关文档"""
        return search(query, top_k=self.top_k)

    def retrieve_as_context(self, query: str) -> str:
        """检索并格式化为上下文字符串（给 Prompt 用）"""
        results = self.retrieve(query)
        if not results:
            return "（未找到相关资料）"

        context_parts = []
        for i, r in enumerate(results):
            text = r['text'][:MAX_CHARS_PER_DOC]
            context_parts.append(
                f"[参考资料 {i+1}] (来源: {r['source']})\n{text}"
            )
        return "\n\n".join(context_parts)


if __name__ == "__main__":
    retriever = ProductRetriever(top_k=3)

    queries = [
        "iPhone 15 有几种颜色？",
        "MacBook Pro 价格多少？",
        "怎么退货？",
    ]

    for q in queries:
        print(f"\n{'='*50}")
        print(f"🔍 查询: {q}")
        print('='*50)
        context = retriever.retrieve_as_context(q)
        print(context)
