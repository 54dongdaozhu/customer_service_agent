"""订单专家 Agent"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
# 依赖包：langgraph langchain_core

from src.tools.order_tools import query_order, track_shipping, list_user_orders


def create_order_agent(llm):
    """
    创建订单专家 Agent

    职责：
    - 查询单笔订单详情
    - 追踪物流状态
    - 查询某用户所有订单

    Args:
        llm: 语言模型实例

    Returns:
        编译好的订单专家 Agent
    """
    system_prompt = """你是一名专业的订单客服专员，熟悉订单系统和物流查询。

你的职责：
1. 根据订单号查询订单详情
2. 查询订单的物流轨迹和当前状态
3. 查询用户名下的所有订单

工作原则：
- 用户提供订单号时，先调用 query_order 查询
- 用户问物流时，调用 track_shipping 查询
- 用户问"我的订单"时，先询问用户名，再调用 list_user_orders
- 回答要清晰，列出关键信息
- 如果订单不存在，友好提示用户检查订单号

你只负责订单相关问题，产品咨询由产品专家负责，投诉由投诉专家负责。
"""

    agent = create_react_agent(
        model=llm,
        tools=[query_order, track_shipping, list_user_orders],
        prompt=system_prompt,
        name="order_expert",
    )

    return agent


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.model import get_llm

    llm = get_llm(temperature=0.1)   # 订单查询用低温度，更精准
    agent = create_order_agent(llm)

    test_cases = [
        "帮我查一下订单 O001",
        "O004 这个订单的物流到哪里了？",
        "张三有哪些订单？",
        "O999 这个订单在哪？",   # 不存在的订单
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print('='*60)

        resp = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        print(f"💬 回答: {resp['messages'][-1].content}")