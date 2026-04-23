"""测试复杂的多 Agent 协作场景"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.utils.model import get_llm
from src.agents.supervisor import build_customer_service_system


def run_complex_test(supervisor, question: str, description: str):
    """运行复杂场景测试"""
    print(f"\n{'='*60}")
    print(f"🔬 场景: {description}")
    print(f"❓ 问题: {question}")
    print('='*60)

    # 流式输出，看清每一步
    print("📡 执行过程：")
    for chunk in supervisor.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values"
    ):
        last_msg = chunk["messages"][-1]
        # 只打印 AI 决策和工具调用，不打印工具结果（太长）
        msg_type = type(last_msg).__name__
        if msg_type == "AIMessage":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"  🔧 [{last_msg.name if hasattr(last_msg, 'name') else 'supervisor'}] 调用: {tc['name']}")
            elif last_msg.content:
                print(f"\n💬 最终回答:\n{last_msg.content}")


def main():
    llm = get_llm(temperature=0.3)
    supervisor = build_customer_service_system(llm)

    complex_cases = [
        (
            "我叫陈伟，我订单O001的iPhone 15收到了但屏幕有坏点，要投诉，顺便问下怎么申请退货",
            "混合场景：投诉 + 产品退货政策"
        ),
        (
            "查一下订单 O004 到哪了，那个MacBook Pro什么时候能到",
            "混合场景：订单查询 + 产品信息"
        ),
        (
            "张三名下有几个订单？每个订单状态是什么？",
            "复杂查询：用户所有订单"
        ),
    ]

    for question, description in complex_cases:
        run_complex_test(supervisor, question, description)


if __name__ == "__main__":
    main()