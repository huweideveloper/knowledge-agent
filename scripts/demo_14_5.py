"""阶段 14.5：演示生成答案是否得到检索资料支持。"""

import sys
from pathlib import Path


# 增加于阶段 14.5：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.faithfulness import evaluate_faithfulness
from app.evaluation.golden_dataset import GoldenRecord, load_golden_dataset
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.5：创建带正文、文档 ID 和页码的固定检索证据。
def _chunk(document_id: str, page: int, content: str) -> RetrievedChunk:
    """创建一条用于 Faithfulness 验收的检索证据。

    实现方式：将正文和来源信息同时写入 RetrievedChunk 与 Metadata，模拟检索器返回
    的完整证据；该函数不访问 Qdrant，保证 Faithfulness Demo 可重复运行。

    参数：
        document_id: 证据所属业务文档版本 ID，必须为非空字符串。
        page: 证据所在页码，必须为正整数。
        content: 证据正文，必须为非空字符串。

    返回：
        RetrievedChunk：包含正文、文档 ID、页码和 Metadata 的固定证据对象。

    异常：
        AssertionError：Demo 传入的文档 ID、页码或正文不符合预期时抛出。
    """
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
        },
    )


# 增加于阶段 14.5：构造两条有正确来源支持和一条来源错误的固定证据。
def _build_retrieval_results(
    records: tuple[GoldenRecord, ...],
) -> dict[str, list[RetrievedChunk]]:
    """为 Golden Dataset 生成可验证 Faithfulness 的证据映射。

    实现方式：第一题和第三题的证据文档、页码及正文均匹配题库，第二题正文虽包含
    生成答案但文档 ID 故意错误；正式评测器必须因此只判定两题得到资料支持。

    参数：
        records: 按 14.1 题库顺序排列的 GoldenRecord 元组，必须正好包含三条记录。

    返回：
        dict[str, list[RetrievedChunk]]：问题到检索证据列表的映射。

    异常：
        AssertionError：Demo 题库数量或记录字段不符合预期时抛出。
    """
    assert len(records) == 3
    first, second, third = records
    return {
        first.question: [_chunk(first.expected_document, first.expected_page, first.expected_answer)],
        second.question: [_chunk("wrong_document", second.expected_page, second.expected_answer)],
        third.question: [_chunk(third.expected_document, third.expected_page, third.expected_answer)],
    }


# 增加于阶段 14.5：运行 Faithfulness 评测并打印答案、证据和汇总结果。
def run_demo() -> None:
    """读取 Golden Dataset 并演示生成答案是否有正确资料支持。

    实现方式：加载 14.1 题库，将期望答案作为固定 LLM 生成答案，并构造两条正确来源
    证据及一条来源错误证据，调用正式 evaluate_faithfulness()；最后打印答案、证据
    正文、文档/页码 Metadata、逐题支持判断和平均 Faithfulness，不访问外部服务。

    参数：
        无入参；使用项目默认 Golden Dataset、三条固定生成答案和三条固定证据。

    返回：
        无返回值；评测输入、关键来源匹配结果、Faithfulness 和验收结论通过标准输出打印。

    异常：
        RuntimeError：题库数量、证据支持关系或 Faithfulness 结果不符合预期时抛出。
    """
    records = load_golden_dataset()
    retrieval_results = _build_retrieval_results(records)
    generated_answers = {record.question: record.expected_answer for record in records}
    report = evaluate_faithfulness(records, generated_answers, retrieval_results)
    expected_scores = (1.0, 0.0, 1.0)
    if (
        report.total_questions != 3
        or report.scores != expected_scores
        or report.supported_questions != (records[0].question, records[2].question)
        or report.unsupported_questions != (records[1].question,)
        or report.faithfulness_score != 2 / 3
    ):
        raise RuntimeError("Faithfulness 验收失败：答案支持关系或指标统计不符合预期")

    print("=== 阶段 14.5 Faithfulness Demo ===")
    print(f"Golden Dataset 题目数: {report.total_questions}")
    print("输入字段: LLM 生成答案 + RetrievedChunk 正文/Metadata + 期望文档页码")
    print("评分方式: 生成答案标准化文本必须出现在正确文档和页码的证据正文中")
    for index, record in enumerate(records, start=1):
        evidence = retrieval_results[record.question][0]
        supported = record.question in report.supported_questions
        print(f"样本 {index} 问题: {record.question}")
        print(f"  LLM 生成答案: {generated_answers[record.question]}")
        print(f"  证据正文: {evidence.content}")
        print(f"  证据文档: {evidence.metadata['document_id']}")
        print(f"  证据页码: {evidence.metadata['page']}")
        print(f"样本 {index} 证据支持: {'是' if supported else '否'}")
    print(f"支持答案数: {len(report.supported_questions)}")
    print(f"不支持答案数: {len(report.unsupported_questions)}")
    print(f"Faithfulness: {report.faithfulness_score:.2%}")
    print("关键处理结果: 第二题正文虽匹配答案，但文档 ID 错误，因此不计为资料支持")
    print("Faithfulness 验收: 通过")


# 增加于阶段 14.5：提供 Faithfulness Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
