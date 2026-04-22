"""模型初始化工具"""
from langchain.chat_models import init_chat_model
# 依赖包：langchain

from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


def get_llm(temperature: float = 0.7, **kwargs):
    """
    获取 LLM 实例

    Args:
        temperature: 随机性 0-1，越大越发散
        **kwargs: 其他传递给模型的参数

    Returns:
        ChatModel 实例
    """
    return init_chat_model(
        model=MODEL_NAME,
        model_provider="openai",
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=temperature,
        **kwargs,
    )
