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

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 填入 OPENAI_API_KEY 和 LANGSMITH_API_KEY
```

### 3. 下载嵌入模型
```bash
python scripts/download_model.py
```

### 4. 构建向量知识库（只需运行一次）
```bash
python scripts/build_vectordb.py
```

### 5. 启动 Web UI
```bash
streamlit run app.py
```

打开浏览器访问：http://localhost:8501

## 项目进度

- [x] 阶段一：环境搭建 + 模型接入
- [x] 阶段二：构建 RAG 产品知识库
- [x] 阶段三：开发工具层
- [x] 阶段四：构建单个专家 Agent
- [x] 阶段五：构建监督者架构
- [x] 阶段六：加入记忆机制
- [x] 阶段七：Web 界面 + LangSmith
