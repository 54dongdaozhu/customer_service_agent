"""测试各 Agent 的流式输出（看每步思考）"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.utils.model import get_llm
from src.agents.product_agent import create_product_agent
from src.agents.order_agent import create_order_agent
from src.agents.complaint_agent import create_complaint_agent


def test_agent_with_stream(agent, question: str, agent_name: str):
    """用流式模式测试 Agent，可以看到每一步的思考过程"""
    print(f"\n{'='*60}")
    print(f"🤖 [{agent_name}]")
    print(f"❓ 问题: {question}")
    print('='*60)

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values"
    ):
        # 只打印最后一条消息（最新的那一步）
        last_msg = chunk["messages"][-1]
        last_msg.pretty_print()


def main():
    llm = get_llm(temperature=0.3)

    # ===== 测试产品专家 =====
    print("\n" + "🌟" * 20)
    print("测试产品专家 Agent")
    print("🌟" * 20)

    product_agent = create_product_agent(llm)
    test_agent_with_stream(
        product_agent,
        "iPhone 15 有几种颜色，最便宜的多少钱？",
        "产品专家"
    )

    # ===== 测试订单专家 =====
    print("\n" + "🌟" * 20)
    print("测试订单专家 Agent")
    print("🌟" * 20)

    order_agent = create_order_agent(llm)
    test_agent_with_stream(
        order_agent,
        "查一下订单 O001，顺便看看物流到哪里了",
        "订单专家"
    )

    # ===== 测试投诉专家 =====
    print("\n" + "🌟" * 20)
    print("测试投诉专家 Agent")
    print("🌟" * 20)

    complaint_agent = create_complaint_agent(llm)
    test_agent_with_stream(
        complaint_agent,
        "我叫陈七，订单O001物流超慢，我要投诉！",
        "投诉专家"
    )


if __name__ == "__main__":
    main()