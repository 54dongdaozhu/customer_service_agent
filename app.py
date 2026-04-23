"""
智能客服系统 - Web UI
运行方式: streamlit run app.py
"""
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 加载环境变量（LangSmith 在 import 模型前就要配置好）
load_dotenv()

from src.utils.model import get_llm
from src.agents.supervisor import build_customer_service_system
from src.memory.checkpointer import make_config

# ============================================================
# 页面基础配置
# ============================================================
st.set_page_config(
    page_title="智能客服系统",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# 自定义样式
# ============================================================
st.markdown("""
<style>
/* 主标题 */
.main-title {
    text-align: center;
    color: #1F4E79;
    font-size: 2rem;
    font-weight: bold;
    padding: 1rem 0;
}
/* Agent 思考过程的小字 */
.agent-trace {
    font-size: 0.8rem;
    color: #888;
    padding: 4px 8px;
    background: #f5f5f5;
    border-left: 3px solid #2E75B6;
    margin: 4px 0;
}
/* 欢迎消息 */
.welcome-box {
    background: linear-gradient(135deg, #E8F4FD, #DBEAFE);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #BFDBFE;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 初始化：模型和 Supervisor（只初始化一次）
# ============================================================
@st.cache_resource
def init_system():
    """初始化客服系统（缓存，只初始化一次）"""
    llm = get_llm(temperature=0.3)
    supervisor = build_customer_service_system(llm)
    return supervisor


# ============================================================
# Session State 初始化
# ============================================================
def init_session():
    """初始化会话状态"""
    if "session_id" not in st.session_state:
        # 每次刷新页面生成新的 session_id
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

    if "messages" not in st.session_state:
        # 聊天历史（用于 UI 显示）
        st.session_state.messages = []

    if "show_trace" not in st.session_state:
        # 是否显示 Agent 思考过程
        st.session_state.show_trace = False


# ============================================================
# 侧边栏
# ============================================================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### 🤖 智能客服系统")
        st.markdown("---")

        # 当前会话信息
        st.markdown("**📋 当前会话**")
        st.code(st.session_state.session_id, language=None)

        # 新建会话按钮
        if st.button("🔄 新建会话", use_container_width=True):
            st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        # 显示思考过程开关
        st.markdown("**⚙️ 设置**")
        st.session_state.show_trace = st.toggle(
            "显示 Agent 思考过程",
            value=st.session_state.show_trace
        )

        st.markdown("---")

        # 系统能力介绍
        st.markdown("**🎯 我能帮您**")
        st.markdown("""
- 📱 **产品咨询**
  价格、参数、颜色、功能
- 📦 **订单查询**
  订单状态、物流追踪
- 😤 **投诉处理**
  创建工单、问题反馈
""")

        st.markdown("---")

        # 快捷提问
        st.markdown("**💡 试试问我**")
        quick_questions = [
            "iPhone 15 有哪些颜色？",
            "查一下订单 O001",
            "O004 的物流到哪了？",
            "我要投诉物流太慢",
        ]

        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                st.session_state.pending_input = q
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#888;font-size:0.8rem;'>"
            "Powered by LangChain + LangGraph"
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# 主聊天区域
# ============================================================
def render_chat():
    """渲染聊天界面"""

    # 欢迎消息（只在没有对话时显示）
    if not st.session_state.messages:
        st.markdown("""
<div class="welcome-box">
    <h3 style="margin:0;color:#1F4E79;">👋 您好！我是智能客服助手</h3>
    <p style="margin:0.5rem 0 0 0;color:#374151;">
    我可以帮您解答产品问题、查询订单状态、处理投诉建议。<br>
    请问有什么可以帮到您的？
    </p>
</div>
""", unsafe_allow_html=True)

    # 显示历史消息
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        trace = message.get("trace", [])

        with st.chat_message(role):
            # 如果开启了思考过程显示，先显示 trace
            if role == "assistant" and st.session_state.show_trace and trace:
                for t in trace:
                    st.markdown(
                        f'<div class="agent-trace">🔧 {t}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown(content)


# ============================================================
# 处理用户输入
# ============================================================
def process_input(supervisor, user_input: str):
    """处理用户输入"""

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    config = make_config(st.session_state.session_id)
    trace_logs = []

    with st.chat_message("assistant"):
        trace_placeholder = st.empty()
        response_placeholder = st.empty()
        full_response = ""

        # ⭐ 记录上一次的消息数量，用于找出新增的消息
        prev_message_count = 0

        try:
            for chunk in supervisor.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if not messages:
                    continue

                # ⭐ 遍历所有新增的消息（不只看最后一条）
                new_messages = messages[prev_message_count:]
                prev_message_count = len(messages)

                for msg in new_messages:
                    msg_type = type(msg).__name__

                    if msg_type == "AIMessage":
                        # 捕获工具调用（思考过程）
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            agent_name = getattr(msg, "name", "supervisor")
                            for tc in msg.tool_calls:
                                log = f"[{agent_name}] 调用工具: {tc['name']}"
                                if tc.get("args"):
                                    args_str = str(tc["args"])[:60]
                                    log += f"  参数: {args_str}"
                                trace_logs.append(log)

                        # 捕获最终文字回复
                        elif msg.content and not getattr(msg, "tool_calls", None):
                            full_response = msg.content

                # 实时更新显示
                if st.session_state.show_trace and trace_logs:
                    trace_html = "".join([
                        f'<div class="agent-trace">🔧 {t}</div>'
                        for t in trace_logs
                    ])
                    trace_placeholder.markdown(trace_html, unsafe_allow_html=True)

                if full_response:
                    response_placeholder.markdown(full_response)

            if not full_response:
                full_response = "抱歉，暂时无法处理您的请求。"
                response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ 系统错误：{str(e)[:100]}"
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "trace": trace_logs,
    })


# ============================================================
# 主函数
# ============================================================
def main():
    # 初始化
    init_session()
    supervisor = init_system()

    # 渲染侧边栏
    render_sidebar()

    # 主标题
    st.markdown(
        '<div class="main-title">🤖 企业级智能客服系统</div>',
        unsafe_allow_html=True
    )

    # 渲染聊天区域
    render_chat()

    # 处理快捷提问（从侧边栏点击的）
    if "pending_input" in st.session_state:
        pending = st.session_state.pop("pending_input")
        process_input(supervisor, pending)
        st.rerun()

    # 处理用户手动输入
    user_input = st.chat_input("请输入您的问题...")
    if user_input:
        process_input(supervisor, user_input)
        st.rerun()


if __name__ == "__main__":
    main()