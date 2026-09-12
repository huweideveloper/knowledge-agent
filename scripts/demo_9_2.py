"""阶段 9.2：演示候选 Chunk 被限制为最多五个。"""

import sys
from pathlib import Path


# 增加于阶段 9.2：支持从项目根目录直接运行 Context 长度控制 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.context_builder import build_context, limit_chunks
from app.retrieval.vector_retriever import search
from app.security import User


# 增加于阶段 9.2：定义真实检索演示使用的 Query 和候选数量。
QUERY = "上海住宿报销标准"
CANDIDATE_COUNT = 7


# 修改于阶段 12.3：为真实 Context 数量控制 Demo 提供经过校验的当前用户。
DEMO_USER = User(id="u001", department="engineering", role="employee")
MAX_CONTEXT_CHUNKS = 5


# 增加于阶段 9.2：执行候选检索、数量控制并打印受控 Context。
def run_demo() -> None:
    """召回七个候选 Chunk 并展示最多保留五个的结果。

    实现方式：调用 search() 获取按相似度排序的七个结果，再用 limit_chunks() 保留
    前五个，最后调用 build_context() 打印实际送入后续 Prompt 的资料文本。

    参数：
        无入参；使用预设 Query、候选数量 7 和 Context 上限 5。

    返回：
        无返回值；候选数量、处理变化和受控 Context 通过标准输出打印。

    异常：
        RuntimeError: Qdrant 返回的候选数量不足七个时抛出。
        TypeError、ValueError: 检索或数量控制参数不符合约束时抛出。
    """
    candidates = search(query=QUERY, top_k=CANDIDATE_COUNT, user=DEMO_USER)
    if len(candidates) != CANDIDATE_COUNT:
        raise RuntimeError(
            f"Context 长度控制验收失败：实际候选 {len(candidates)} 条"
        )
    retained = limit_chunks(candidates, max_chunks=MAX_CONTEXT_CHUNKS)
    context = build_context(retained)

    print("=== 阶段 9.2 Context 长度控制 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"输入候选数量: {len(candidates)}")
    print(f"最大保留数量: {MAX_CONTEXT_CHUNKS}")
    print(f"关键处理结果: 保持检索顺序并截取前 {MAX_CONTEXT_CHUNKS} 个 Chunk")
    print(f"受控 Chunk 数量: {len(retained)}")
    print("最终输出:")
    print(context)
    print("Context 长度控制验收: 通过")


# 增加于阶段 9.2：提供 Context 长度控制 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
