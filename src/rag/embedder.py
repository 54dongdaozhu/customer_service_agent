"""嵌入模块"""
from langchain_huggingface import HuggingFaceEmbeddings
# 依赖包：langchain_huggingface sentence-transformers

from config.settings import EMBEDDING_MODEL_PATH


_embed_model = None


def get_embedder() -> HuggingFaceEmbeddings:
    """获取嵌入模型单例（避免重复加载）"""
    global _embed_model
    if _embed_model is None:
        print(f"🔧 加载嵌入模型: {EMBEDDING_MODEL_PATH}")
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"},  # Mac 用 cpu，有 GPU 可改 "cuda"
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ 嵌入模型加载完成")
    return _embed_model


if __name__ == "__main__":
    # 测试嵌入
    embedder = get_embedder()

    # 测试单文本嵌入
    text = "iPhone 15 有几种颜色？"
    vec = embedder.embed_query(text)
    print(f"\n向量维度: {len(vec)}")
    print(f"前 5 个数字: {vec[:5]}")

    # 测试语义相似度
    import numpy as np
    v1 = embedder.embed_query("iPhone 15 的价格")
    v2 = embedder.embed_query("iPhone 多少钱")
    v3 = embedder.embed_query("量子力学")

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\niPhone价格 vs iPhone多少钱: {cosine(v1, v2):.4f}  (应该很高)")
    print(f"iPhone价格 vs 量子力学:   {cosine(v1, v3):.4f}  (应该很低)")
