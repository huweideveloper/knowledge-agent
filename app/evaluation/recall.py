"""阶段 14.2：计算 Retrieval Recall@K。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.evaluation.golden_dataset import GoldenRecord
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.2：定义 Recall@K 的可打印评测报告。
@dataclass(frozen=True)
class RecallAtKReport:
    """保存 Top K 命中题数、未命中题目和召回率。"""

    k: int
    total_questions: int
    hit_questions: tuple[str, ...]
    missed_questions: tuple[str, ...]
    recall_at_k: float


# 增加于阶段 14.2：根据 Golden Dataset 和检索结果计算 Recall@K。
def evaluate_recall_at_k(
    golden_records: Iterable[GoldenRecord],
    retrieved_results: Mapping[str, Iterable[RetrievedChunk]],
    k: int = 5,
) -> RecallAtKReport:
    """计算正确文档出现在每道题前 K 条结果中的比例。

    实现方式：校验 K、GoldenRecord 序列和按问题索引的 RetrievedChunk 结果，对每条
    题目只查看前 K 条 Chunk；优先读取 Chunk Metadata 的 document_id，并回退到 source
    字段，与 GoldenRecord.expected_document 精确比较，最后返回命中题目和召回率。
    缺少某道题的检索结果按未命中处理，不改变 Golden Dataset 或检索结果。

    参数：
        golden_records: 14.1 Golden Dataset 记录，可迭代且不能为空，问题必须唯一。
        retrieved_results: 问题到检索结果序列的映射；每条结果必须是完整
            RetrievedChunk，结果顺序必须与检索排名一致。
        k: 评估的 Top K，必须为正整数，默认 5。

    返回：
        RecallAtKReport：包含 K 值、总题数、命中问题、未命中问题及 0 到 1 之间的
            recall_at_k；题库中所有问题都会进入统计。

    异常：
        TypeError: K、题库、结果映射或其中元素类型不符合接口约束时抛出。
        ValueError: K 不为正数、题库为空、问题重复或结果 Metadata 不是对象时抛出。
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if isinstance(golden_records, (str, bytes)):
        raise TypeError("golden_records must be an iterable of GoldenRecord")
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
        if record.question in questions:
            raise ValueError(f"golden_records contains duplicate question: {record.question}")
        questions.add(record.question)

    hit_questions: list[str] = []
    missed_questions: list[str] = []
    for record in records:
        raw_results = retrieved_results.get(record.question, ())
        if isinstance(raw_results, (str, bytes)):
            raise TypeError("retrieved results must be an iterable of RetrievedChunk")
        try:
            result_list = list(raw_results)
        except TypeError as error:
            raise TypeError(
                "retrieved results must be an iterable of RetrievedChunk"
            ) from error
        top_results = result_list[:k]
        if any(not isinstance(chunk, RetrievedChunk) for chunk in top_results):
            raise TypeError("retrieved results must contain only RetrievedChunk objects")

        hit = False
        for chunk in top_results:
            if not isinstance(chunk.metadata, Mapping):
                raise ValueError("RetrievedChunk.metadata must be an object")
            document_id = chunk.metadata.get("document_id") or chunk.source
            if document_id == record.expected_document:
                hit = True
                break
        if hit:
            hit_questions.append(record.question)
        else:
            missed_questions.append(record.question)

    return RecallAtKReport(
        k=k,
        total_questions=len(records),
        hit_questions=tuple(hit_questions),
        missed_questions=tuple(missed_questions),
        recall_at_k=len(hit_questions) / len(records),
    )
