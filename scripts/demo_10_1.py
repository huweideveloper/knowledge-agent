"""阶段 10.1：演示每个检索结果保留 Citation 来源字段。"""

import sys
from pathlib import Path


# 增加于阶段 10.1：支持从项目根目录直接运行来源保留 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_retriever import RetrievedChunk, search


# 增加于阶段 10.1：定义真实检索来源字段演示的 Query 和数量。
QUERY = "上海住宿报销标准"
TOP_K = 5


# 增加于阶段 10.1：检索并打印每条结果的 source、page、chunk_id。
def run_demo() -> None:
    """从 Qdrant 召回结果并展示 Citation 所需的三项来源信息。

    实现方式：调用 search() 获取标准 RetrievedChunk，逐条校验 source、page 和
    chunk_id 均非空，再按排名打印三项字段和正文摘要；不让 LLM 参与来源生成。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=5。

    返回：
        无返回值；检索输入、来源字段和验收结论通过标准输出打印。

    异常：
        RuntimeError: Qdrant 返回结果不足五条，或任一结果缺少来源字段时抛出。
        TypeError、ValueError: 检索参数不符合约束时由 search() 抛出。
    """
    results = search(query=QUERY, top_k=TOP_K)
    if len(results) != TOP_K:
        raise RuntimeError(f"来源保留验收失败：实际返回 {len(results)} 条结果")
    if not all(
        isinstance(result, RetrievedChunk)
        and isinstance(result.source, str)
        and result.source.strip()
        and result.page is not None
        and isinstance(result.chunk_id, str)
        and result.chunk_id.strip()
        for result in results
    ):
        raise RuntimeError("来源保留验收失败：存在结果缺少 source/page/chunk_id")

    print("=== 阶段 10.1 检索来源保留 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"返回结果数量: {len(results)}")
    print("关键处理结果: 每个 RetrievedChunk 均保留 source、page、chunk_id")
    print("最终输出:")
    for rank, result in enumerate(results, start=1):
        print(
            f"#{rank} source={result.source} page={result.page} "
            f"chunk_id={result.chunk_id}"
        )
        print(f"content={result.content.strip()[:120]}")
    print("来源保留验收: 通过")


# 增加于阶段 10.1：提供来源保留 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
