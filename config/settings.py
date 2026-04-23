"""项目配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ========== 模型配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:free")

# ========== 嵌入模型配置 ==========
EMBEDDING_MODEL_PATH = str(PROJECT_ROOT / "models" / "bge-base-zh-v1.5")

# ========== 向量数据库配置 ==========
MILVUS_DB_PATH = str(PROJECT_ROOT / "data" / "milvus.db")
COLLECTION_NAME = "product_knowledge"
VECTOR_DIM = 768  # bge-base-zh 的维度

# ========== 业务数据配置 ==========
MOCK_DB_PATH = str(PROJECT_ROOT / "data" / "mock_db.json")
TICKETS_DB_PATH = str(PROJECT_ROOT / "data" / "tickets.json")
# ========== LangSmith 配置 ==========
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "customer-service-agent")