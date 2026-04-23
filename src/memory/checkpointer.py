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
def get_session_history(supervisor, thread_id: str) -> list:
    """
    查看某个会话的完整对话历史

    Args:
        supervisor: 监督者系统
        thread_id: 会话ID

    Returns:
        消息列表
    """
    config = make_config(thread_id)
    state = supervisor.get_state(config)
    return state.values.get("messages", [])


def print_session_history(supervisor, thread_id: str):
    """打印某个会话的对话历史"""
    messages = get_session_history(supervisor, thread_id)

    print(f"\n📋 会话 [{thread_id}] 共 {len(messages)} 条消息：")
    print('─' * 50)

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            print(f"👤 用户: {msg.content[:100]}")
        elif msg_type == "AIMessage" and msg.content:
            print(f"🤖 AI: {msg.content[:100]}")
    print('─' * 50)