"""阶段 14.6：演示 Citation Accuracy 的计算。"""

import sys
from pathlib import Path


# 增加于阶段 14.6：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.citation_accuracy import evaluate_citation_accuracy
from app.evaluation.golden_dataset import GoldenRecord, load_golden_dataset
from app.rag.citations import CitedAnswer, build_cited_answer
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.6：创建带唯一 Chunk ID、来源和页码的固定检索结果。
def _chunk(
    chunk_id: str,
    document_id: str,
    page: int,
    content: str,
) -> RetrievedChunk:
    """创建一条可映射为可信 Citation 的检索 Chunk。

    实现方式：将 Chunk ID、业务文档 ID、页码和正文写入标准 RetrievedChunk，供已有
    build_cited_answer() 从真实结果生成 Citation；该函数不访问 Qdrant，保证 Demo 可
    重复运行。

    参数：
        chunk_id: Chunk 的唯一标识，必须为非空字符串。
        document_id: Chunk 所属业务文档版本 ID，必须为非空字符串。
        page: Chunk 所在页码，必须为正整数。
        content: Chunk 正文，必须为非空字符串。

    返回：
        RetrievedChunk：包含正文、来源、页码、Metadata 和 Chunk ID 的检索结果。

    异常：
        AssertionError：Demo 传入的 Chunk 字段不符合预期时抛出。
    """
    assert chunk_id
    assert document_id
    assert page > 0
    assert content
    return RetrievedChunk(
        content=content,
        score=0.95,
        source=document_id,
        page=page,
        metadata={
            "document_id": document_id,
            "source": document_id,
            "page": page,
            "chunk_id": chunk_id,
        },
        chunk_id=chunk_id,
    )


# 增加于阶段 14.6：构造两条正确 Citation 和一条错误来源 Citation。
def _build_cited_answers(
    records: tuple[GoldenRecord, ...],
) -> dict[str, CitedAnswer]:
    """为 Golden Dataset 生成可验证 Citation Accuracy 的可信回答。

    实现方式：第一题和第三题把 LLM citation ID 映射到题库规定的文档页码，第二题
    将 citation ID 映射到 wrong_document；所有 Citation 均先经过正式
    build_cited_answer() 从真实 Chunk 生成，评测器再检查最终 source/page。

    参数：
        records: 按 14.1 题库顺序排列的 GoldenRecord 元组，必须正好包含三条记录。

    返回：
        dict[str, CitedAnswer]：问题到已映射可信回答的结果字典。

    异常：
        AssertionError：Demo 题库数量或记录字段不符合预期时抛出。
        ValueError、TypeError、pydantic.ValidationError：Citation 映射输入不合法时
            由 build_cited_answer() 透传。
    """
    assert len(records) == 3
    first, second, third = records
    first_chunk = _chunk("chunk_001", first.expected_document, first.expected_page, first.expected_answer)
    second_chunk = _chunk("chunk_002", "wrong_document", second.expected_page, second.expected_answer)
    third_chunk = _chunk("chunk_003", third.expected_document, third.expected_page, third.expected_answer)
    return {
        first.question: build_cited_answer(
            {"answer": first.expected_answer, "citations": ["chunk_001"]},
            [first_chunk],
        ),
        second.question: build_cited_answer(
            {"answer": second.expected_answer, "citations": ["chunk_002"]},
            [second_chunk],
        ),
        third.question: build_cited_answer(
            {"answer": third.expected_answer, "citations": ["chunk_003"]},
            [third_chunk],
        ),
    }


# 增加于阶段 14.6：运行 Citation Accuracy 评测并打印输入、映射结果和汇总指标。
def run_demo() -> None:
    """读取 Golden Dataset 并演示 Citation 是否指向正确来源。

    实现方式：加载 14.1 题库，模拟 LLM 返回 citation ID，调用已有
    build_cited_answer() 从固定检索 Chunk 生成可信 Citation，再调用正式
    evaluate_citation_accuracy() 比较文档和页码；最后打印 citation ID、映射后的
    Citation、逐题准确判断和总体准确率，不访问外部模型或检索服务。

    参数：
        无入参；使用项目默认 Golden Dataset 和三条固定 Citation 映射结果。

    返回：
        无返回值；评测输入、关键来源页码匹配结果、Citation Accuracy 和验收结论
        通过标准输出打印。

    异常：
        RuntimeError：题库数量、Citation 映射或准确率结果不符合预期时抛出。
    """
    records = load_golden_dataset()
    cited_answers = _build_cited_answers(records)
    report = evaluate_citation_accuracy(records, cited_answers)
    expected_scores = (1.0, 0.0, 1.0)
    if (
        report.total_questions != 3
        or report.scores != expected_scores
        or report.accurate_citations != 2
        or report.total_citations != 3
        or report.citation_accuracy != 2 / 3
    ):
        raise RuntimeError("Citation Accuracy 验收失败：引用来源或指标统计不符合预期")

    print("=== 阶段 14.6 Citation Accuracy Demo ===")
    print(f"Golden Dataset 题目数: {report.total_questions}")
    print("输入字段: LLM Citation ID + 真实 RetrievedChunk Metadata + 期望文档页码")
    print("处理方式: 先由 Chunk ID 映射可信 Citation，再比较 source/page")
    for index, record in enumerate(records, start=1):
        cited_answer = cited_answers[record.question]
        citation = cited_answer.citations[0]
        score = report.scores[index - 1]
        print(f"样本 {index} 问题: {record.question}")
        print(f"  LLM Citation ID: {citation.chunk_id}")
        print(f"样本 {index} Citation: {citation.label}")
        print(f"样本 {index} 引用准确: {'是' if score == 1.0 else '否'}")
        print(f"样本 {index} Citation Accuracy: {score:.2%}")
    print(f"正确 Citation 数: {report.accurate_citations}")
    print(f"总 Citation 数: {report.total_citations}")
    print(f"Citation Accuracy: {report.citation_accuracy:.2%}")
    print("关键处理结果: 第二题 Citation 映射到 wrong_document，因此不计为准确引用")
    print("Citation Accuracy 验收: 通过")


# 增加于阶段 14.6：提供 Citation Accuracy Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
