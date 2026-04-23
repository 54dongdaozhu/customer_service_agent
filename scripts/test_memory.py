"""测试记忆机制：对比有无记忆的效果"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.utils.model import get_llm
from src.agents.supervisor import build_customer_service_system
from src.memory.checkpointer import make_config


def chat(supervisor, user_input: str, config: dict) -> str:
    """单次对话"""
    resp = supervisor.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    return resp["messages"][-1].content


def test_without_memory(supervisor):
    """❌ 没有记忆：每次都是全新对话"""
    print("\n" + "🔴" * 20)
    print("测试：没有记忆（每次用不同 thread_id）")
    print("🔴" * 20)

    # 每次用不同 thread_id = 每次全新对话
    print("\n第一轮：")
    reply1 = chat(supervisor, "我叫王伟", make_config("session_A"))
    print(f"👤 用户: 我叫王伟")
    print(f"🤖 客服: {reply1}")

    print("\n第二轮（新 thread_id，失忆）：")
    reply2 = chat(supervisor, "我叫什么名字？", make_config("session_B"))
    print(f"👤 用户: 我叫什么名字？")
    print(f"🤖 客服: {reply2}")
    print("⚠️ 预期：不记得用户叫王伟")


def test_with_memory(supervisor):
    """✅ 有记忆：同一个 thread_id，保持上下文"""
    print("\n" + "🟢" * 20)
    print("测试：有记忆（同一个 thread_id）")
    print("🟢" * 20)

    config = make_config("user_wangwei_001")  # 固定 thread_id

    conversations = [
        "你好，我叫王伟",
        "我想查一下我的订单",
        "订单号是 O001",
        "物流到哪里了？",     # 不再提订单号，靠记忆
        "我叫什么名字？",     # 测试是否记得姓名
    ]

    for user_input in conversations:
        print(f"\n👤 用户: {user_input}")
        reply = chat(supervisor, user_input, config)
        print(f"🤖 客服: {reply}")


def test_multi_user(supervisor):
    """✅ 多用户隔离：不同用户的对话互不干扰"""
    print("\n" + "🔵" * 20)
    print("测试：多用户隔离")
    print("🔵" * 20)

    # 用户A
    config_a = make_config("user_zhangsan")
    print("\n[用户A - 张三]")
    chat(supervisor, "我叫张三，我的订单是 O001", config_a)
    print(f"👤 张三: 我叫张三，我的订单是 O001")

    # 用户B
    config_b = make_config("user_lisi")
    print("\n[用户B - 李四]")
    chat(supervisor, "我叫李四，我的订单是 O002", config_b)
    print(f"👤 李四: 我叫李四，我的订单是 O002")

    # 验证隔离
    print("\n[验证隔离]")
    reply_a = chat(supervisor, "我叫什么名字？我的订单是什么？", config_a)
    print(f"👤 张三问: 我叫什么名字？我的订单是什么？")
    print(f"🤖 客服对张三: {reply_a}")
    print("✅ 预期：回答张三、O001，不会混入李四的信息")

    reply_b = chat(supervisor, "我叫什么名字？我的订单是什么？", config_b)
    print(f"\n👤 李四问: 我叫什么名字？我的订单是什么？")
    print(f"🤖 客服对李四: {reply_b}")
    print("✅ 预期：回答李四、O002，不会混入张三的信息")


def main():
    llm = get_llm(temperature=0.3)
    supervisor = build_customer_service_system(llm)

    test_without_memory(supervisor)
    test_with_memory(supervisor)
    test_multi_user(supervisor)

    print("\n" + "="*60)
    print("✅ 记忆机制测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()