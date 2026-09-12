"""阶段 14.6：计算 Citation 是否指向正确的来源和页码。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.evaluation.golden_dataset import GoldenRecord
from app.rag.citations import Citation, CitedAnswer


# 增加于阶段 14.6：定义 Citation Accuracy 评测报告。
@dataclass(frozen=True)
class CitationAccuracyReport:
    """保存逐题引用准确率、准确引用数和总体准确率。"""

    total_questions: int
    scores: tuple[float, ...]
    accurate_questions: tuple[str, ...]
    inaccurate_questions: tuple[str, ...]
    accurate_citations: int
    total_citations: int
    citation_accuracy: float


# 增加于阶段 14.6：根据可信 CitedAnswer 和 Golden Dataset 计算 Citation Accuracy。
def evaluate_citation_accuracy(
    golden_records: Iterable[GoldenRecord],
    cited_answers: Mapping[str, CitedAnswer],
) -> CitationAccuracyReport:
    """计算每条 Citation 是否指向题库规定的正确文档和页码。

    实现方式：校验 Golden Dataset 和问题到 CitedAnswer 的映射，逐题检查每个 Citation
    的 source/page；source 与 expected_document、page 与 expected_page 均匹配时计为
    准确引用。每题分数为准确引用数除以该题引用总数，没有回答或没有 Citation 的题
    计为 0，最后以全部准确引用除以全部 Citation 数得到总体 Citation Accuracy。
    CitedAnswer 应先由 build_cited_answer() 从真实检索 Chunk 映射得到，本函数不
    信任 LLM 自行编造的来源字段。

    参数：
        golden_records: 14.1 Golden Dataset 记录，可迭代且不能为空，问题必须唯一，
            并提供正确文档 ID 和页码。
        cited_answers: 问题到已完成可信 Citation 映射的 CitedAnswer；缺少某题按
            没有准确引用计分。

    返回：
        CitationAccuracyReport：包含总题数、按题库顺序排列的逐题分数、准确/不准确
            问题、准确 Citation 数、总 Citation 数和 0 到 1 之间的总体准确率。

    异常：
        TypeError：题库、答案映射或其中元素类型不符合接口约束时抛出。
        ValueError：题库为空、问题重复、期望来源字段不合法或 Citation 字段不完整时
            抛出。
    """
    if isinstance(golden_records, (str, bytes)):
        raise TypeError("golden_records must be an iterable of GoldenRecord")
    if not isinstance(cited_answers, Mapping):
        raise TypeError("cited_answers must be a mapping")
    try:
        records = list(golden_records)
    except TypeError as error:
        raise TypeError("golden_records must be an iterable of GoldenRecord") from error
    if not records:
        raise ValueError("golden_records must not be empty")

    questions: set[str] = set()
    for record in records:
        if not isinstance(record, GoldenRecord):
            raise TypeError("golden_records must contain only GoldenRecord objects")
        if not isinstance(record.question, str) or not record.question.strip():
            raise ValueError("GoldenRecord.question must not be empty")
        if not isinstance(record.expected_document, str) or not record.expected_document.strip():
            raise ValueError("GoldenRecord.expected_document must not be empty")
        if (
            isinstance(record.expected_page, bool)
            or not isinstance(record.expected_page, int)
            or record.expected_page <= 0
        ):
            raise ValueError("GoldenRecord.expected_page must be a positive integer")
        if record.question in questions:
            raise ValueError(f"golden_records contains duplicate question: {record.question}")
        questions.add(record.question)

    scores: list[float] = []
    accurate_questions: list[str] = []
    inaccurate_questions: list[str] = []
    accurate_citations = 0
    total_citations = 0
    for record in records:
        cited_answer = cited_answers.get(record.question)
        if cited_answer is None:
            scores.append(0.0)
            inaccurate_questions.append(record.question)
            continue
        if not isinstance(cited_answer, CitedAnswer):
            raise TypeError("cited_answers values must be CitedAnswer objects")

        citations = cited_answer.citations
        if not isinstance(citations, list):
            raise TypeError("CitedAnswer.citations must be a list")
        accurate_for_question = 0
        for citation in citations:
            if not isinstance(citation, Citation):
                raise ValueError("CitedAnswer.citations must contain Citation objects")
            if not isinstance(citation.source, str) or not citation.source.strip():
                raise ValueError("Citation.source must not be empty")
            if citation.page is None or isinstance(citation.page, bool):
                raise ValueError("Citation.page must be valid")
            if (
                citation.source.strip() == record.expected_document.strip()
                and str(citation.page).strip() == str(record.expected_page)
            ):
                accurate_for_question += 1

        total_citations += len(citations)
        accurate_citations += accurate_for_question
        score = (
            accurate_for_question / len(citations)
            if citations
            else 0.0
        )
        scores.append(score)
        if score == 1.0:
            accurate_questions.append(record.question)
        else:
            inaccurate_questions.append(record.question)

    total_questions = len(records)
    return CitationAccuracyReport(
        total_questions=total_questions,
        scores=tuple(scores),
        accurate_questions=tuple(accurate_questions),
        inaccurate_questions=tuple(inaccurate_questions),
        accurate_citations=accurate_citations,
        total_citations=total_citations,
        citation_accuracy=(accurate_citations / total_citations if total_citations else 0.0),
    )
