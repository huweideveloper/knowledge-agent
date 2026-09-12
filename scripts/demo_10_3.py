"""阶段 10.3：演示 Citation 是否真实支持答案内容。"""

import sys
from pathlib import Path


# 增加于阶段 10.3：支持从项目根目录直接运行 Citation 内容验证 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.citations import CitedAnswer, build_cited_answer, verify_citation_support
from app.retrieval.vector_retriever import search


# 增加于阶段 10.3：定义真实检索和支持关系验证的输入。
QUERY = "上海住宿报销标准"
TOP_K = 5


# 增加于阶段 10.3：执行支持答案与错误答案的 Citation 内容验证。
def run_demo() -> None:
    """验证真实检索 Chunk 支持 600 元/晚答案并拒绝 800 元/晚反例。

    实现方式：从 Qdrant 获取结果，使用 10.2 映射函数生成合法 CitedAnswer，再用
    verify_citation_support() 检查数字事实；随后只篡改答案金额而保留同一 Citation，
    验证程序能识别不受引用内容支持的错误答案。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=5。

    返回：
        无返回值；支持事实、错误事实和验收结论通过标准输出打印。

    异常：
        RuntimeError: 没有可引用结果、支持答案未通过或错误答案未被拒绝时抛出。
        TypeError、ValueError: 检索或 Citation 数据不符合约束时抛出。
    """
    chunks = search(query=QUERY, top_k=TOP_K)
    if not chunks or not chunks[0].chunk_id:
        raise RuntimeError("Citation 内容验证失败：没有可引用的检索 Chunk")

    supported_answer = build_cited_answer(
        {
            "answer": "普通员工去上海出差的酒店上限为 600 元/晚。",
            "citations": [chunks[0].chunk_id],
        },
        chunks,
    )
    supported_result = verify_citation_support(supported_answer, chunks)

    unsupported_answer = CitedAnswer(
        answer="普通员工去上海出差的酒店上限为 800 元/晚。",
        citations=supported_answer.citations,
    )
    unsupported_result = verify_citation_support(unsupported_answer, chunks)
    if not supported_result.supported or unsupported_result.supported:
        raise RuntimeError("Citation 内容验证失败：支持关系判断错误")

    print("=== 阶段 10.3 Citation 内容验证 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"引用 Chunk: {supported_result.cited_chunk_ids[0]}")
    print(f"匹配事实: {', '.join(supported_result.matched_facts)}")
    print("支持答案验证: 通过")
    print(f"错误答案缺失事实: {', '.join(unsupported_result.missing_facts)}")
    print("错误答案验证: 拒绝")
    print("Citation 内容验证验收: 通过")


# 增加于阶段 10.3：提供 Citation 内容验证 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
