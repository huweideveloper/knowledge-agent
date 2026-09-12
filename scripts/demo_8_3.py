"""阶段 8.3：在终端按规定格式打印 Top 5 检索结果。"""

import sys
from pathlib import Path


# 增加于阶段 8.3：支持从项目根目录直接运行 Top 5 输出 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_retriever import search
from app.security import User


# 增加于阶段 8.3：定义 Top 5 输出 Demo 的输入 Query 和召回数量。
QUERY = "上海住宿报销标准"
TOP_K = 5


# 修改于阶段 12.3：为真实检索 Demo 提供经过校验的当前用户。
DEMO_USER = User(id="u001", department="engineering", role="employee")


# 增加于阶段 8.3：执行检索并按文档格式打印 Top 5。
def run_demo() -> None:
    """检索上海住宿标准并在终端逐条打印 Top 5 结果。

    实现方式：调用阶段 8.2 的 search() 获取 RetrievedChunk 列表，确认实际召回
    五条结果后，按排名、score、来源、页码、内容的固定顺序输出；不改变检索
    结果，也不在本任务中接入 LLM 或额外重排。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=5。

    返回：
        无返回值；输入、Top 5 结果及验收结论通过标准输出打印。

    异常：
        RuntimeError: Qdrant 返回结果少于五条时抛出。
        TypeError、ValueError: 检索参数不符合约束时由 search() 抛出。
    """
    results = search(query=QUERY, top_k=TOP_K, user=DEMO_USER)
    if len(results) != TOP_K:
        raise RuntimeError(f"Top 5 验收失败：实际返回 {len(results)} 条结果")

    print("=== 阶段 8.3 Top 5 检索结果 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"输入 top_k: {TOP_K}")
    print(f"关键处理结果: 已按相似度召回 {len(results)} 条 RetrievedChunk")
    print("最终输出:")
    for rank, result in enumerate(results, start=1):
        print(f"#{rank} score={result.score:.6f}")
        print()
        print("来源：")
        print(result.source or "未知来源")
        print()
        print("页码：")
        print(result.page if result.page is not None else "未知页码")
        print()
        print("内容：")
        print(result.content.strip())
        if rank != len(results):
            print()

    print("Top 5 验收: 通过")
    print("最终输出: 终端已打印 Top 5 RetrievedChunk")


# 增加于阶段 8.3：提供 Top 5 输出 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
