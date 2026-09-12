"""阶段 14.5：计算生成答案是否得到检索资料支持。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.evaluation.answer_correctness import normalize_answer
from app.evaluation.golden_dataset import GoldenRecord
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.5：定义 Faithfulness 评测报告。
@dataclass(frozen=True)
class FaithfulnessReport:
    """保存逐题资料支持分数、支持题数和平均 Faithfulness。"""

    total_questions: int
    scores: tuple[float, ...]
    supported_questions: tuple[str, ...]
    unsupported_questions: tuple[str, ...]
    faithfulness_score: float


# 增加于阶段 14.5：根据证据正文和来源 Metadata 计算 Faithfulness。
# ponytail: V1 仅检查答案标准化文本是否出现在正确文档页码的证据中；复杂多跳声明留给后续升级。
def evaluate_faithfulness(
    golden_records: Iterable[GoldenRecord],
    generated_answers: Mapping[str, str],
    retrieved_results: Mapping[str, Iterable[RetrievedChunk]],
) -> FaithfulnessReport:
    """计算生成答案是否被正确来源的检索资料直接支持。

    实现方式：校验 Golden Dataset、生成答案和按问题索引的 RetrievedChunk 结果；对每
    道题寻找文档 ID 和页码均匹配 expected_document/expected_page 的证据，并检查
    生成答案的标准化文本是否包含在证据正文中。存在这样的证据记为 1.0，否则记为
    0.0，最后返回逐题分数、支持/不支持问题和平均 Faithfulness。该 V1 规则不调用
    外部 LLM，不判断同义改写或多跳推理，专门隔离“答案是否有资料支持”这一指标。

    参数：
        golden_records: 14.1 Golden Dataset 记录，可迭代且不能为空，问题必须唯一，
            并提供正确文档 ID 和页码。
        generated_answers: 问题到最终生成答案的映射；缺少某题答案按不受支持计分。
        retrieved_results: 问题到检索证据序列的映射；每条结果必须是完整
            RetrievedChunk，且顺序与检索返回顺序一致。

    返回：
        FaithfulnessReport：包含总题数、按题库顺序排列的 0.0/1.0 分数、支持问题、
            不支持问题和 0 到 1 之间的平均 faithfulness_score。

    异常：
        TypeError：题库、答案映射、结果映射或其中元素类型不符合接口约束时抛出。
        ValueError：题库为空、问题重复、期望来源字段不合法、证据 Metadata 不是对象
            或证据正文为空时抛出。
    """
    if isinstance(golden_records, (str, bytes)):
        raise TypeError("golden_records must be an iterable of GoldenRecord")
    if not isinstance(generated_answers, Mapping):
        raise TypeError("generated_answers must be a mapping")
    if not isinstance(retrieved_results, Mapping):
        raise TypeError("retrieved_results must be a mapping")
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
    supported_questions: list[str] = []
    unsupported_questions: list[str] = []
    for record in records:
        generated_answer = generated_answers.get(record.question, "")
        if not isinstance(generated_answer, str):
            raise TypeError("generated_answers values must be strings")
        normalized_answer = normalize_answer(generated_answer)
        raw_results = retrieved_results.get(record.question, ())
        if isinstance(raw_results, (str, bytes)):
            raise TypeError("retrieved results must be an iterable of RetrievedChunk")
        try:
            result_list = list(raw_results)
        except TypeError as error:
            raise TypeError(
                "retrieved results must be an iterable of RetrievedChunk"
            ) from error

        supported = False
        for chunk in result_list:
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("retrieved results must contain only RetrievedChunk objects")
            if not isinstance(chunk.metadata, Mapping):
                raise ValueError("RetrievedChunk.metadata must be an object")
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise ValueError("RetrievedChunk.content must not be empty")
            document_id = chunk.metadata.get("document_id") or chunk.source
            page = chunk.metadata.get("page_label") or chunk.metadata.get("page") or chunk.page
            source_matches = document_id == record.expected_document
            page_matches = str(page).strip() == str(record.expected_page)
            content_matches = normalized_answer and normalized_answer in normalize_answer(
                chunk.content
            )
            if source_matches and page_matches and content_matches:
                supported = True

        score = float(supported)
        scores.append(score)
        if supported:
            supported_questions.append(record.question)
        else:
            unsupported_questions.append(record.question)

    total_questions = len(records)
    return FaithfulnessReport(
        total_questions=total_questions,
        scores=tuple(scores),
        supported_questions=tuple(supported_questions),
        unsupported_questions=tuple(unsupported_questions),
        faithfulness_score=sum(scores) / total_questions,
    )
