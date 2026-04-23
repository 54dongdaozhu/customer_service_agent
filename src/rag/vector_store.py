"""向量存储模块（Milvus）"""
from typing import List

from pymilvus import MilvusClient, DataType
from langchain_core.documents import Document
# 依赖包：pymilvus langchain_core

from config.settings import MILVUS_DB_PATH, COLLECTION_NAME, VECTOR_DIM
from src.rag.embedder import get_embedder


def build_schema():
    """构建 Milvus Collection 的 Schema"""
    return (
        MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        .add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        .add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        .add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
        .add_field(field_name="source", datatype=DataType.VARCHAR, max_length=500)
    )


def build_index_params():
    """构建索引参数"""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",  # 余弦相似度
    )
    return index_params


def get_client() -> MilvusClient:
    """获取 Milvus 客户端"""
    return MilvusClient(uri=MILVUS_DB_PATH)


def create_collection(recreate: bool = False):
    """
    创建向量集合

    Args:
        recreate: 如果已存在，是否重建
    """
    client = get_client()

    if client.has_collection(COLLECTION_NAME):
        if recreate:
            print(f"⚠️ 删除已有集合: {COLLECTION_NAME}")
            client.drop_collection(COLLECTION_NAME)
        else:
            print(f"✅ 集合已存在: {COLLECTION_NAME}")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=build_schema(),
        index_params=build_index_params(),
    )
    print(f"✅ 集合创建成功: {COLLECTION_NAME}")


def insert_documents(chunks: List[Document]):
    """
    把切分后的 Chunks 插入向量库

    Args:
        chunks: Document 列表
    """
    client = get_client()
    embedder = get_embedder()

    # 批量计算向量
    print(f"🔧 正在计算 {len(chunks)} 个块的向量...")
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    # 组装数据
    data = [
        {
            "vector": vec,
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", "unknown"),
        }
        for chunk, vec in zip(chunks, vectors)
    ]

    # 插入
    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"✅ 插入成功: {result['insert_count']} 条")


def search(query: str, top_k: int = 3) -> List[dict]:
    """
    根据查询文本进行向量检索

    Args:
        query: 用户查询
        top_k: 返回前 K 个结果

    Returns:
        检索结果列表
    """
    client = get_client()
    embedder = get_embedder()

    # 嵌入查询
    query_vector = embedder.embed_query(query)

    # 检索
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "source"],
        limit=top_k,
    )

    # 解析结果
    return [
        {
            "text": hit["entity"]["text"],
            "source": hit["entity"]["source"],
            "score": hit["distance"],
        }
        for hit in results[0]
    ]


if __name__ == "__main__":
    # 测试检索
    results = search("iPhone 15 有几种颜色？")
    print(f"\n检索结果:\n")
    for i, r in enumerate(results):
        print(f"--- 结果 {i+1} (相似度: {r['score']:.4f}) ---")
        print(f"来源: {r['source']}")
        print(f"内容: {r['text'][:200]}...")
        print()
