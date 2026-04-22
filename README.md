# 🤖 企业级智能客服系统

基于 LangChain 构建的智能客服系统，支持：
- 📚 产品问答（RAG）
- 📦 订单查询
- 📝 投诉处理
- 🎯 多 Agent 协作

## 技术栈

- LangChain / LangGraph
- Milvus 向量数据库
- bge-base-zh 嵌入模型
- Streamlit Web UI

## 快速开始

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd customer_service_agent
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
复制 `.env.example` 为 `.env`，填入你的 API Key

### 5. 运行测试
```bash
python scripts/test_hello.py
```

## 项目进度

- [x] 阶段一：环境搭建 + 模型接入
- [ ] 阶段二：构建 RAG 产品知识库
- [ ] 阶段三：开发工具层
- [ ] 阶段四：构建单个专家 Agent
- [ ] 阶段五：构建监督者架构
- [ ] 阶段六：加入记忆机制
- [ ] 阶段七：Web 界面 + LangSmith
