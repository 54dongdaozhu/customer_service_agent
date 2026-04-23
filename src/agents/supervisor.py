"""监督者 Agent（主管）"""
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
# 依赖包：langgraph-supervisor langchain_core

from src.agents.product_agent import create_product_agent
from src.agents.order_agent import create_order_agent
from src.agents.complaint_agent import create_complaint_agent
from src.memory.checkpointer import get_memory_saver


def build_customer_service_system(llm):
    """
    构建完整的智能客服系统

    架构：
        Supervisor（主管）
            ├── product_expert（产品专家）
            ├── order_expert（订单专家）
            └── complaint_expert（投诉专家）

    Args:
        llm: 语言模型实例

    Returns:
        编译好的监督者系统
    """

    # ========== Step 1：创建三个专家 Agent ==========
    product_agent  = create_product_agent(llm)
    order_agent    = create_order_agent(llm)
    complaint_agent = create_complaint_agent(llm)

    # ========== Step 2：设计主管提示词（核心！） ==========
    supervisor_prompt = """你是一名经验丰富的客服主管，负责协调以下三位专家：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👩‍💼 团队成员介绍：

📱 product_expert（产品专家）
   擅长：产品参数、价格、功能、退换货政策、保修政策
   触发词：多少钱、颜色、参数、型号、功能、退货、保修、怎么用

📦 order_expert（订单专家）
   擅长：查询订单状态、物流追踪、查看历史订单
   触发词：订单、物流、快递、发货、到哪了、状态

😤 complaint_expert（投诉专家）
   擅长：处理投诉、创建工单、安抚用户情绪
   触发词：投诉、不满、太慢、有问题、要退款、太差了、气死了

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 路由决策原则：

1. 【产品问题】→ 派 product_expert
   例："iPhone 15 有几种颜色？"、"MacBook 多少钱？"

2. 【订单/物流问题】→ 派 order_expert
   例："查一下订单O001"、"我的快递到哪了？"

3. 【投诉/不满】→ 派 complaint_expert
   例："我要投诉！"、"物流太慢了"、"产品有质量问题"

4. 【混合问题】→ 按优先级派遣
   - 有强烈情绪 → 先派 complaint_expert 安抚
   - 产品+订单 → 先派 order_expert 查订单，再派 product_expert 答产品问题

5. 【无法判断】→ 礼貌询问用户具体需求

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 工作流程：
1. 分析用户意图
2. 选择合适的专家
3. 等待专家返回结果
4. 整合结果，用友好的语气回复用户
5. 如有需要，继续派遣其他专家
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 记忆原则（重要！）：
- 记住用户的姓名，之后的对话中主动使用
- 记住用户提过的订单号，不要让用户重复说
- 记住用户的问题背景，保持对话连贯性
- 如果用户说"刚才那个"、"上面说的"，结合历史理解

"""

    # ========== Step 3：创建 Supervisor ==========
    # ⭐ 关键：传入 checkpointer 开启记忆
    memory = get_memory_saver()

    supervisor = create_supervisor(
        agents=[product_agent, order_agent, complaint_agent],
        model=llm,
        prompt=supervisor_prompt,
    ).compile(
        checkpointer=memory  # ⭐ 加在 compile 里
    )

    return supervisor

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.model import get_llm

    llm = get_llm(temperature=0.3)
    supervisor = build_customer_service_system(llm)

    # 快速冒烟测试
    test = "iPhone 15 多少钱？"
    print(f"测试: {test}")
    resp = supervisor.invoke({
        "messages": [HumanMessage(content=test)]
    })
    print(f"回答: {resp['messages'][-1].content}")