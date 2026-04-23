"""工单相关工具"""
import json
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
# 依赖包：langchain_core

from config.settings import TICKETS_DB_PATH


def _load_tickets() -> dict:
    """加载工单数据"""
    if not Path(TICKETS_DB_PATH).exists():
        return {"tickets": []}
    with open(TICKETS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tickets(data: dict):
    """保存工单数据"""
    with open(TICKETS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@tool
def create_ticket(
    user_name: str,
    issue_type: str,
    description: str,
    order_id: str = "",
) -> str:
    """
    创建客服投诉/售后工单。

    Args:
        user_name: 用户姓名
        issue_type: 问题类型，可选值：物流问题、产品质量、退换货、服务投诉、其他
        description: 问题详细描述
        order_id: 相关订单号（可选，如果是针对具体订单的问题）

    Returns:
        创建成功后返回工单号和工单详情
    """
    data = _load_tickets()

    # 生成工单号
    ticket_count = len(data["tickets"]) + 1
    ticket_id = f"T{ticket_count:04d}"

    # 创建工单
    new_ticket = {
        "ticket_id": ticket_id,
        "user_name": user_name,
        "issue_type": issue_type,
        "description": description,
        "order_id": order_id or "无",
        "status": "待处理",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    data["tickets"].append(new_ticket)
    _save_tickets(data)

    return (
        f"✅ 工单创建成功\n"
        f"工单号: {ticket_id}\n"
        f"用户: {user_name}\n"
        f"类型: {issue_type}\n"
        f"关联订单: {new_ticket['order_id']}\n"
        f"描述: {description}\n"
        f"创建时间: {new_ticket['created_at']}\n"
        f"状态: 待处理（客服会在24小时内联系您）"
    )


@tool
def query_ticket(ticket_id: str) -> str:
    """
    查询工单状态和详情。

    Args:
        ticket_id: 工单号，格式如 "T0001"

    Returns:
        工单详情字符串
    """
    data = _load_tickets()

    for ticket in data["tickets"]:
        if ticket["ticket_id"] == ticket_id:
            return (
                f"📋 工单详情\n"
                f"工单号: {ticket['ticket_id']}\n"
                f"用户: {ticket['user_name']}\n"
                f"类型: {ticket['issue_type']}\n"
                f"描述: {ticket['description']}\n"
                f"关联订单: {ticket['order_id']}\n"
                f"状态: {ticket['status']}\n"
                f"创建时间: {ticket['created_at']}"
            )

    return f"❌ 未找到工单 {ticket_id}"


if __name__ == "__main__":
    print("=" * 50)
    print("测试 create_ticket:")
    print("=" * 50)
    result = create_ticket.invoke({
        "user_name": "张三",
        "issue_type": "物流问题",
        "description": "订单 O001 已经一周了还没收到，物流状态一直不更新",
        "order_id": "O001",
    })
    print(result)

    print("\n" + "=" * 50)
    print("测试 query_ticket:")
    print("=" * 50)
    print(query_ticket.invoke({"ticket_id": "T0001"}))
