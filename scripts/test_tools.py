"""综合测试：让 LLM 调用所有工具"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, ToolMessage

from src.utils.model import get_llm
from src.tools.order_tools import query_order, track_shipping, list_user_orders
from src.tools.ticket_tools import create_ticket, query_ticket
from src.tools.product_tools import search_product_info


# 收集所有工具
ALL_TOOLS = [
    query_order,
    track_shipping,
    list_user_orders,
    create_ticket,
    query_ticket,
    search_product_info,
]

# 工具名 → 工具对象的映射（用于执行时查找）
TOOLS_MAP = {tool.name: tool for tool in ALL_TOOLS}


def run_with_tools(user_input: str):
    """让 LLM 自主判断是否调用工具，并执行完整流程"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [HumanMessage(content=user_input)]
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_input}")
    print('='*60)

    # 循环：直到 LLM 不再需要调用工具
    max_iterations = 5
    for i in range(max_iterations):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # 没有工具调用 → 给出最终答案
            print(f"\n🤖 最终回答:\n{ai_msg.content}")
            return

        # 执行每个工具调用
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"\n🔧 调用工具: {tool_name}")
            print(f"   参数: {tool_args}")

            # 执行工具
            tool_obj = TOOLS_MAP[tool_name]
            tool_result = tool_obj.invoke(tool_args)
            print(f"   结果: {tool_result[:100]}...")

            # 把结果包装成 ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    print("⚠️ 达到最大迭代次数")


def main():
    test_cases = [
        "帮我查一下订单 O001 的状态",
        "订单 O004 的物流到哪了？",
        "张三有哪些订单？",
        "iPhone 15 有哪些颜色？",
        "我要投诉：订单 O001 的物流太慢了，请帮我创建工单，我叫王强",
    ]

    for question in test_cases:
        run_with_tools(question)
        print("\n")


if __name__ == "__main__":
    main()
