"""文档切分模块"""
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# 依赖包：langchain-text-splitters langchain_core


def split_documents(
    docs: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    切分文档为小块

    Args:
        docs: Document 列表
        chunk_size: 每块最大字符数
        chunk_overlap: 块间重叠字符数

    Returns:
        切分后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    print(f"✅ 切分完成：{len(docs)} 个文档 → {len(chunks)} 个块")
    return chunks


if __name__ == "__main__":
    # 测试切分
    from src.rag.loader import load_directory

    docs = load_directory("knowledge_base")
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)

    print(f"\n前 3 个块预览：\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} (长度: {len(chunk.page_content)}) ---")
        print(chunk.page_content)
        print()
