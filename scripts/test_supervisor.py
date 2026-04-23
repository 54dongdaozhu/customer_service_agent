"""测试监督者路由准确性"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage

from src.utils.model import get_llm
from src.agents.supervisor import build_customer_service_system


def test_routing(supervisor, question: str, expected_agent: str):
    """
    测试路由是否正确

    Args:
        supervisor: 监督者系统
        question: 用户问题
        expected_agent: 期望被调用的 Agent 名称
    """
    print(f"\n{'─'*60}")
    print(f"❓ 问题: {question}")
    print(f"🎯 期望路由到: {expected_agent}")
    print('─'*60)

    resp = supervisor.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # 分析消息历史，找出调用了哪个 Agent
    messages = resp["messages"]
    called_agents = []
    for msg in messages:
        if hasattr(msg, "name") and msg.name in [
            "product_expert", "order_expert", "complaint_expert"
        ]:
            if msg.name not in called_agents:
                called_agents.append(msg.name)

    # 判断路由是否正确
    is_correct = expected_agent in called_agents
    status = "✅ 路由正确" if is_correct else "❌ 路由错误"

    print(f"📡 实际路由到: {called_agents}")
    print(f"📊 结果: {status}")
    print(f"💬 最终回答: {messages[-1].content[:150]}...")

    return is_correct


def main():
    llm = get_llm(temperature=0.1)  # 低温度让路由更稳定
    supervisor = build_customer_service_system(llm)

    # ===== 路由测试用例 =====
    test_cases = [
        # (问题, 期望路由到)
        ("iPhone 15 有哪些颜色？",          "product_expert"),
        ("MacBook Pro 16寸多少钱？",        "product_expert"),
        ("买了产品怎么退货？",              "product_expert"),
        ("帮我查一下订单 O001",             "order_expert"),
        ("O004 的快递到哪了？",             "order_expert"),
        ("张三有哪些订单？",                "order_expert"),
        ("我要投诉，物流太慢了！我叫赵明", "complaint_expert"),
        ("产品质量有问题，我要投诉",        "complaint_expert"),
        ("我叫刘强，O002 的 Mac 屏幕坏了，要投诉", "complaint_expert"),
    ]

    # 执行测试
    correct = 0
    total = len(test_cases)

    for question, expected in test_cases:
        if test_routing(supervisor, question, expected):
            correct += 1

    # 打印总结
    print(f"\n{'='*60}")
    print(f"📊 路由测试结果: {correct}/{total} 正确")
    accuracy = correct / total * 100
    print(f"🎯 准确率: {accuracy:.1f}%")

    if accuracy >= 80:
        print("✅ 路由准确率达标（≥80%），阶段五通过！")
    else:
        print("⚠️ 路由准确率不足，需要优化 Supervisor 的 prompt")
    print('='*60)

# 在 scripts/test_supervisor.py 末尾加一个函数

def simulate_conversation(supervisor):
    """模拟完整客服对话"""
    print(f"\n{'='*60}")
    print("🎭 模拟完整客服对话")
    print('='*60)

    # 构建对话历史
    conversation = [
        "你好",
        "我想了解一下 iPhone 15 的价格",
        "128GB 的最便宜，我想查一下我之前的订单",
        "订单号是 O001",
        "好的，那个物流怎么这么慢！我要投诉，我叫周明",
    ]

    messages = []
    for user_input in conversation:
        print(f"\n👤 用户: {user_input}")
        messages.append(HumanMessage(content=user_input))

        resp = supervisor.invoke({"messages": messages})
        ai_reply = resp["messages"][-1].content
        print(f"🤖 客服: {ai_reply}")

        # 把 AI 回复加入历史（保持多轮对话）
        messages = resp["messages"]

if __name__ == "__main__":
    main()