"""阶段 10.2：演示结构化 LLM 输出到可信 Citation 的程序映射。"""

import json
import sys
from pathlib import Path


# 增加于阶段 10.2：支持从项目根目录直接运行可信 Citation Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.citations import build_cited_answer
from app.retrieval.vector_retriever import search


# 增加于阶段 10.2：定义真实检索和结构化回答 Demo 的输入。
QUERY = "上海住宿报销标准"
TOP_K = 5


# 增加于阶段 10.2：执行结构化 Citation 映射并打印可信来源。
def run_demo() -> None:
    """从真实检索结果构造结构化回答并映射为可信 Citation。

    实现方式：先调用 search() 获取带真实来源的 Chunk，再模拟 LLM 只返回 answer
    和第一个真实 chunk_id；build_cited_answer() 根据 Chunk Map 重新读取 source/page
    并生成用户可读标签，本 Demo 不让模型直接提供文件名或页码。

    参数：
        无入参；使用预设 Query“上海住宿报销标准”和 top_k=5。

    返回：
        无返回值；结构化输入、映射结果和验收结论通过标准输出打印。

    异常：
        RuntimeError: Qdrant 没有返回可引用结果时抛出。
        TypeError、ValueError、pydantic.ValidationError: 输入或来源 Metadata 不完整时抛出。
    """
    chunks = search(query=QUERY, top_k=TOP_K)
    if not chunks or not chunks[0].chunk_id:
        raise RuntimeError("Citation 映射验收失败：没有可引用的检索 Chunk")

    llm_output = {
        "answer": "普通员工上海住宿上限为 600 元/晚。",
        "citations": [chunks[0].chunk_id],
    }
    cited_answer = build_cited_answer(llm_output, chunks)

    print("=== 阶段 10.2 可信 Citation 映射 Demo ===")
    print(f"输入 Query: {QUERY}")
    print(f"检索 Chunk 数量: {len(chunks)}")
    print(f"LLM 结构化输出: {json.dumps(llm_output, ensure_ascii=False)}")
    print("关键处理结果: 程序根据 chunk_id 从真实 Chunk Map 读取 source/page")
    print(f"最终回答: {cited_answer.answer}")
    for citation in cited_answer.citations:
        print(
            f"映射 Citation: {citation.label} "
            f"(chunk_id={citation.chunk_id}, source={citation.source}, page={citation.page})"
        )
    print("Citation 映射验收: 通过")


# 增加于阶段 10.2：提供可信 Citation Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
