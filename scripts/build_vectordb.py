"""一键构建向量数据库"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import load_directory
from src.rag.splitter import split_documents
from src.rag.vector_store import create_collection, insert_documents


def main():
    print("=" * 50)
    print("🚀 开始构建产品知识库向量数据库")
    print("=" * 50)

    # Step 1: 创建集合（如果已存在则删除重建）
    create_collection(recreate=True)

    # Step 2: 加载文档
    docs = load_directory("knowledge_base")

    # Step 3: 切分
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)

    # Step 4: 插入向量库
    insert_documents(chunks)

    print("\n" + "=" * 50)
    print("✅ 向量数据库构建完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
