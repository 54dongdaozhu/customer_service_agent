"""投诉专家 Agent"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
# 依赖包：langgraph langchain_core

from src.tools.ticket_tools import create_ticket, query_ticket
from src.tools.order_tools import query_order


def create_complaint_agent(llm):
    """
    创建投诉专家 Agent

    职责：
    - 倾听用户投诉
    - 查询相关订单了解情况
    - 创建投诉工单
    - 查询已有工单状态

    Args:
        llm: 语言模型实例

    Returns:
        编译好的投诉专家 Agent
    """
    system_prompt = """你是一名专业的客服投诉专员，善于安抚用户情绪并高效处理投诉。

你的职责：
1. 耐心倾听用户投诉，表达同理心
2. 如果用户提到订单号，查询订单了解背景
3. 为用户创建投诉工单，记录问题
4. 告知用户工单号和后续处理流程

工作原则：
- 首先表达歉意和理解，让用户感受到被重视
- 收集必要信息：用户姓名、问题类型、详细描述、订单号（如有）
- 问题类型选项：物流问题、产品质量、退换货、服务投诉、其他
- 创建工单前确认用户提供了姓名
- 工单创建后告知工单号，承诺24小时内跟进
- 语气要诚恳、专业，不推卸责任

你只负责投诉处理，产品咨询和订单查询由其他专家负责。
"""

    agent = create_react_agent(
        model=llm,
        tools=[create_ticket, query_ticket, query_order],
        prompt=system_prompt,
        name="complaint_expert",
    )

    return agent


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.model import get_llm
    from langchain_core.messages import HumanMessage

    llm = get_llm(temperature=0.5)
    agent = create_complaint_agent(llm)

    test_cases = [
        "我要投诉！订单O001的物流一直不动，我叫李明",
        "我买的MacBook有问题，屏幕有坏点，我叫王芳，订单是O002",
        "查一下我的工单 T0001 处理到哪一步了",
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print('='*60)

        resp = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        print(f"💬 回答: {resp['messages'][-1].content}")