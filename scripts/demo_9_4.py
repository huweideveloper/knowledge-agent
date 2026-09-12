"""阶段 9.4：演示完整 RAG Pipeline 的五段式数据流。"""

import sys
from pathlib import Path


# 增加于阶段 9.4：支持从项目根目录直接运行 RAG Pipeline Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.llm import get_default_chat_model
from app.rag.pipeline import RagPipeline


# 增加于阶段 9.4：定义端到端流程演示问题。
QUESTION = "普通员工去上海出差，酒店最多报多少？"


# 增加于阶段 9.4：执行并打印五段式 RAG Pipeline 的关键结果。
def run_demo() -> None:
    """使用真实检索和默认聊天模型运行一次完整 RAG 流程。

    实现方式：创建按环境选择的 DeepSeek 或本地 grounded fallback，注入
    RagPipeline，执行上海住宿问题并打印模型类型、召回数量、Context 预览和回答。

    参数：
        无入参；使用需求文档规定的上海出差住宿问题。

    返回：
        无返回值；Pipeline 各阶段和最终回答通过标准输出打印。

    异常：
        RuntimeError: Qdrant、Embedding 或配置的远程模型不可用时抛出。
        TypeError、ValueError: Pipeline 输入不符合约束时抛出。
    """
    model = get_default_chat_model()
    result = RagPipeline(llm=model).invoke(QUESTION)
    if not result.chunks or not result.answer:
        raise RuntimeError("RAG Pipeline 验收失败：没有资料或回答")

    print("=== 阶段 9.4 RAG Pipeline Demo ===")
    print(f"输入 Query: {QUESTION}")
    print("流程: query → retriever → context builder → prompt → LLM")
    print(f"模型适配器: {type(model).__name__}")
    print(f"检索 Chunk 数量: {len(result.chunks)}")
    print(f"Context 字符数: {len(result.context)}")
    print("模型回答:")
    print(result.answer)
    print("RAG Pipeline 验收: 通过")


# 增加于阶段 9.4：提供 RAG Pipeline Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
