"""文档加载模块"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
# 依赖包：langchain_community langchain_core


def load_document(file_path: str) -> List[Document]:
    """
    根据文件后缀自动选择加载器

    Args:
        file_path: 文件路径

    Returns:
        Document 列表
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".md":
        loader = UnstructuredMarkdownLoader(str(path), mode="single")
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    return loader.load()


def load_directory(dir_path: str) -> List[Document]:
    """
    加载整个目录下的所有支持的文档

    Args:
        dir_path: 目录路径

    Returns:
        所有文档的 Document 列表
    """
    all_docs = []
    supported_exts = {".md", ".txt", ".pdf"}

    for file in Path(dir_path).iterdir():
        if file.suffix.lower() in supported_exts:
            print(f"📄 加载: {file.name}")
            docs = load_document(str(file))
            all_docs.extend(docs)

    print(f"✅ 共加载 {len(all_docs)} 个文档")
    return all_docs


if __name__ == "__main__":
    # 测试加载
    docs = load_directory("knowledge_base")
    for doc in docs:
        print(f"\n--- {doc.metadata.get('source')} ---")
        print(doc.page_content[:100] + "...")
