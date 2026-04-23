"""记忆管理模块"""
from langgraph.checkpoint.memory import InMemorySaver
# 依赖包：langgraph

# 全局单例 —— 整个应用共享一个记忆存储
_memory_saver = None


def get_memory_saver() -> InMemorySaver:
    """
    获取内存记忆保存器（单例）

    Returns:
        InMemorySaver 实例
    """
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = InMemorySaver()
    return _memory_saver


def make_config(thread_id: str) -> dict:
    """
    生成对话配置（每个用户/会话用不同 thread_id）

    Args:
        thread_id: 会话唯一标识，如用户ID或会话ID

    Returns:
        config 字典，传给 agent.invoke() 的 config 参数
    """
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }