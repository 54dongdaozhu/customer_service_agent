"""Agent 基础配置"""
from langgraph.prebuilt import create_react_agent
# 依赖包：langgraph

# create_react_agent 是 LangGraph 内置的 ReAct Agent 工厂函数
# 比 LangChain 的 create_agent 更稳定，推荐使用
