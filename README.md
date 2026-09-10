# Enterprise Knowledge Agent

企业内部知识库 Agent 项目。

当前状态：已完成阶段 3.5。后续阶段按
`enterprise-knowledge-agent-implementation-plan-v2.md` 逐步实现。

## 本地环境

要求 Python 3.11 或更高版本。

所有注释都是中文注释。

```bash
source venv/bin/activate
python --version
python main.py
```

当前已添加 PDF Loader 所需的 LangChain、langchain-community 和 pypdf 依赖，
后续阶段按实际需要继续更新 `requirements.txt`。
