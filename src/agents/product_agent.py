"""产品专家 Agent"""
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
# 依赖包：langgraph langchain_core

from src.tools.product_tools import search_product_info


def create_product_agent(llm):
    """
    创建产品专家 Agent

    职责：
    - 回答产品参数、功能、规格等问题
    - 查询产品价格
    - 解答退换货、保修政策
    - 对比不同产品型号

    Args:
        llm: 语言模型实例

    Returns:
        编译好的产品专家 Agent
    """
    system_prompt = """你是一名专业的产品顾问，拥有丰富的产品知识。

你的职责：
1. 回答关于产品的各种问题（参数、价格、功能、颜色等）
2. 解答退换货和保修政策
3. 帮助用户对比不同产品

工作原则：
- 必须先调用 search_product_info 工具搜索产品知识库，再作答
- 回答要准确、简洁、友好
- 如果搜索结果中没有相关信息，诚实告知用户
- 使用适当的 emoji 让回答更生动

你只负责产品相关问题，订单查询和投诉处理由其他专家负责。
"""

    agent = create_react_agent(
        model=llm,
        tools=[search_product_info],
        prompt=system_prompt,
        name="product_expert",     # ⭐ 监督者模式必须要有 name
    )

    return agent


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.model import get_llm

    llm = get_llm(temperature=0.3)
    agent = create_product_agent(llm)

    test_cases = [
        "iPhone 15 有哪些颜色？",
        "MacBook Pro 和 MacBook Air 有什么区别？",
        "买了产品后悔了可以退货吗？",
        "AirPods Pro 多少钱？",
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print('='*60)

        resp = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })
        print(f"💬 回答: {resp['messages'][-1].content}")
