"""项目配置"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ========== 模型配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:free")

# ========== 其他配置（后续阶段补充）==========
# EMBEDDING_MODEL_PATH = ""
# MILVUS_DB_PATH = ""
