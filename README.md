# Enterprise Knowledge Agent

企业内部知识库 Agent 项目。

当前状态：已完成阶段 11.4。后续阶段按
`enterprise-knowledge-agent-implementation-plan-v2.md` 逐步实现。

## 本地环境

要求 Python 3.11 或更高版本。

所有注释都是中文注释。

```bash
source venv/bin/activate
python --version
python main.py
```

当前已添加 PDF Loader、Embedding 和 Qdrant 客户端所需依赖，后续阶段按实际需要
继续更新 `requirements.txt`。

## DeepSeek 配置

项目从根目录 `.env` 读取 DeepSeek 配置，默认模型调用使用
`DEEPSEEK_API_KEY`：

```dotenv
DEEPSEEK_API_KEY=你的 DeepSeek API Key
```

`.env` 已加入 `.gitignore`，不要将真实 Key 提交到 Git。运行 RAG Demo 前无需
额外执行 `export`，代码会自动读取该文件。
