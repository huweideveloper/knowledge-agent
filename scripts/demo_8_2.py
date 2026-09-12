"""阶段 8.2：演示检索结果统一返回 RetrievedChunk 标准对象。"""

import sys
from pathlib import Path


# 增加于阶段 8.2：支持从项目根目录直接运行标准检索对象 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_retriever import RetrievedChunk, search
from app.security import User


# 增加于阶段 8.2：定义标准检索对象 Demo 的输入 Query 和 Top K。
QUERY = "上海住宿报销标准"
TOP_K = 5


# 修改于阶段 12.3：为真实检索 Demo 提供经过校验的当前用户。
DEMO_USER = User(id="u001", department="engineering", role="employee")


# 增加于阶段 8.2：执行检索并输出标准对象的关键字段。
def run_demo() -> None:
    """检索上海住宿标准并展示 RetrievedChunk 的统一字段。

    实现方式：调用向量检索函数生成 Top K 结果，校验结果非空且每条都是
    RetrievedChunk，再打印首条结果的正文、相似度、来源、页码和 Metadata，
    让调用方可以直接观察检索输出的数据结构。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=5。

    返回：
        无返回值；检索输入、结果数量、标准对象和验收结论通过标准输出打印。

    异常：
        RuntimeError: Qdrant 没有召回结果，或返回对象不是 RetrievedChunk 时抛出。
        TypeError、ValueError: 检索参数不符合约束时由 search() 抛出。
    """
    results = search(query=QUERY, top_k=TOP_K, user=DEMO_USER)
    if not results:
        raise RuntimeError("标准对象验收失败：没有召回任何 Chunk")
    if not all(isinstance(result, RetrievedChunk) for result in results):
        raise RuntimeError("标准对象验收失败：结果不是 RetrievedChunk")

    first_result = results[0]
    print("=== 阶段 8.2 标准检索对象 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"输入 top_k: {TOP_K}")
    print(f"返回结果数量: {len(results)}")
    print("标准对象类型: RetrievedChunk")
    print(f"首条结果对象: {first_result!r}")
    print(f"首条 content 长度: {len(first_result.content)}")
    print(f"首条 score: {first_result.score:.6f}")
    print(f"首条 source: {first_result.source}")
    print(f"首条 page: {first_result.page}")
    print(f"首条 metadata keys: {sorted(first_result.metadata)}")
    print("关键处理结果: Query 经过 Embedding 和 Qdrant 检索后转换为统一标准对象")
    print("标准对象验收: 通过")
    print("最终输出: RetrievedChunk 可直接作为后续 RAG Context 的输入对象")


# 增加于阶段 8.2：提供标准检索对象 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
