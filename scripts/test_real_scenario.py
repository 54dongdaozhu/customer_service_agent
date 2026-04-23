"""模拟真实客服完整场景"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.utils.model import get_llm
from src.agents.supervisor import build_customer_service_system
from src.memory.checkpointer import make_config, print_session_history


def run_scenario(supervisor, scenario_name: str,
                 thread_id: str, conversations: list):
    """运行一个客服场景"""
    print(f"\n{'='*60}")
    print(f"🎭 场景: {scenario_name}")
    print('='*60)

    config = make_config(thread_id)

    for user_input in conversations:
        print(f"\n👤 用户: {user_input}")
        resp = supervisor.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        reply = resp["messages"][-1].content
        print(f"🤖 客服: {reply}")

    # 打印会话历史摘要
    print_session_history(supervisor, thread_id)


def main():
    llm = get_llm(temperature=0.3)
    supervisor = build_customer_service_system(llm)

    # ===== 场景一：用户咨询后下单，再查物流 =====
    run_scenario(
        supervisor,
        scenario_name="咨询 → 查订单 → 查物流",
        thread_id="scenario_1",
        conversations=[
            "你好，我叫陈明",
            "iPhone 15 128GB 多少钱？",
            "好的，我之前买了一台，订单号 O001",
            "它现在发货了吗？",           # 不再重复订单号
            "大概什么时候能到？",          # 继续追问
        ]
    )

    # ===== 场景二：用户投诉，需要多个专家协作 =====
    run_scenario(
        supervisor,
        scenario_name="产品问题 → 投诉 → 创建工单",
        thread_id="scenario_2",
        conversations=[
            "我叫刘洋，我的 MacBook 屏幕有问题",
            "订单号是 O002",
            "这个在保修范围内吗？",        # 问产品政策
            "太气了，我要投诉！",          # 情绪升级
            "帮我创建投诉工单",            # 记住之前信息创建工单
        ]
    )


if __name__ == "__main__":
    main()