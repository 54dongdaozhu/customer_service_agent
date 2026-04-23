"""订单相关工具"""
import json
from typing import Optional
from pathlib import Path

from langchain_core.tools import tool
# 依赖包：langchain_core

from config.settings import MOCK_DB_PATH


def _load_db() -> dict:
    """加载模拟数据库"""
    with open(MOCK_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def query_order(order_id: str) -> str:
    """
    根据订单号查询订单详细信息。

    Args:
        order_id: 订单号，格式如 "O001"

    Returns:
        订单详情字符串，包括：商品、数量、金额、状态、下单日期、收货地址等
        如果订单不存在，返回错误提示
    """
    db = _load_db()
    orders = db.get("orders", [])

    for order in orders:
        if order["order_id"] == order_id:
            return (
                f"📦 订单信息\n"
                f"订单号: {order['order_id']}\n"
                f"用户: {order['user_name']}\n"
                f"商品: {order['product']}\n"
                f"数量: {order['quantity']}\n"
                f"金额: ¥{order['amount']}\n"
                f"状态: {order['status']}\n"
                f"下单日期: {order['create_date']}\n"
                f"收货地址: {order['address']}"
            )

    return f"❌ 未找到订单 {order_id}，请检查订单号是否正确"


@tool
def track_shipping(order_id: str) -> str:
    """
    查询订单的物流信息。

    Args:
        order_id: 订单号，格式如 "O001"

    Returns:
        物流详情字符串，包括：快递单号、承运商、当前位置、预计送达时间、物流轨迹
        如果订单未发货或不存在，返回相应提示
    """
    db = _load_db()
    shipping = db.get("shipping", {})

    if order_id not in shipping:
        # 进一步查原因
        orders = db.get("orders", [])
        order = next((o for o in orders if o["order_id"] == order_id), None)
        if not order:
            return f"❌ 未找到订单 {order_id}"
        return f"⚠️ 订单 {order_id} 当前状态为「{order['status']}」，暂无物流信息"

    info = shipping[order_id]
    result = [
        f"🚚 物流信息",
        f"快递单号: {info['tracking_no']}",
        f"承运商: {info['carrier']}",
        f"当前位置: {info['current_location']}",
        f"预计送达: {info['estimated_delivery']}",
        f"\n📍 物流轨迹:",
    ]
    for update in info["updates"]:
        result.append(f"  {update['time']} - {update['location']} - {update['status']}")

    return "\n".join(result)


@tool
def list_user_orders(user_name: str) -> str:
    """
    查询某个用户名下的所有订单。

    Args:
        user_name: 用户姓名，如 "张三"

    Returns:
        该用户所有订单的列表，包括订单号、商品、金额、状态
        如果该用户没有订单，返回相应提示
    """
    db = _load_db()
    orders = db.get("orders", [])

    user_orders = [o for o in orders if o["user_name"] == user_name]

    if not user_orders:
        return f"❌ 用户「{user_name}」名下没有订单"

    result = [f"📋 用户「{user_name}」共有 {len(user_orders)} 笔订单:\n"]
    for o in user_orders:
        result.append(
            f"  - {o['order_id']} | {o['product']} | ¥{o['amount']} | {o['status']}"
        )

    return "\n".join(result)


if __name__ == "__main__":
    # 测试工具
    print("=" * 50)
    print("测试 query_order:")
    print("=" * 50)
    print(query_order.invoke({"order_id": "O001"}))

    print("\n" + "=" * 50)
    print("测试 track_shipping:")
    print("=" * 50)
    print(track_shipping.invoke({"order_id": "O001"}))

    print("\n" + "=" * 50)
    print("测试 list_user_orders:")
    print("=" * 50)
    print(list_user_orders.invoke({"user_name": "张三"}))

    print("\n" + "=" * 50)
    print("测试查不存在的订单:")
    print("=" * 50)
    print(query_order.invoke({"order_id": "O999"}))
