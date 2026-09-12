"""阶段 9.1：演示多个检索 Chunk 拼接为 Context。"""

import sys
from pathlib import Path


# 增加于阶段 9.1：支持从项目根目录直接运行 Context Builder Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.context_builder import build_context
from app.retrieval.vector_retriever import search


# 增加于阶段 9.1：定义用于真实检索演示的 Query 和结果数量。
QUERY = "上海住宿报销标准"
TOP_K = 2


# 增加于阶段 9.1：执行真实检索并打印拼接后的 Context。
def run_demo() -> None:
    """从 Qdrant 召回两个结果并演示 Context Builder 的最终文本。

    实现方式：调用现有 search() 获取真实 RetrievedChunk，使用 build_context() 按
    需求格式拼接，再打印 Query、输入数量、关键处理结果和完整 Context。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=2。

    返回：
        无返回值；验收信息通过标准输出打印。

    异常：
        RuntimeError: Qdrant 未返回恰好两个结果时抛出。
        TypeError、ValueError: 检索参数不符合约束时由 search() 抛出。
    """
    results = search(query=QUERY, top_k=TOP_K)
    if len(results) != TOP_K:
        raise RuntimeError(f"Context Builder 验收失败：实际返回 {len(results)} 条结果")

    context = build_context(results)
    print("=== 阶段 9.1 Context Builder Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"输入 Chunk 数量: {len(results)}")
    print("关键处理结果: 已按检索顺序生成资料编号、来源和正文")
    print("最终输出:")
    print(context)
    print("Context Builder 验收: 通过")


# 增加于阶段 9.1：提供 Context Builder Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
