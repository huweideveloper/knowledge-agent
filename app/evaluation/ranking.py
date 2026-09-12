"""阶段 14.3：计算正确文档排名对应的 MRR 和 Top1 Accuracy。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.evaluation.golden_dataset import GoldenRecord
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.3：定义 MRR / Top1 的不可变评测报告。
@dataclass(frozen=True)
class RankingReport:
    """保存每题正确文档排名及 Top1、MRR 汇总指标。"""

    total_questions: int
    correct_ranks: tuple[int | None, ...]
    top1_hits: int
    top1_accuracy: float
    mrr: float


# 增加于阶段 14.3：根据完整检索排序计算正确文档排名、Top1 和 MRR。
def evaluate_ranking(
    golden_records: Iterable[GoldenRecord],
    retrieved_results: Mapping[str, Iterable[RetrievedChunk]],
) -> RankingReport:
    """计算题库中正确文档的首位命中率和平均倒数排名。

    实现方式：校验 GoldenRecord 序列和按问题索引的 RetrievedChunk 结果，按检索结果
    原有顺序从第 1 位开始寻找每道题的 expected_document；找不到时记录 None。Top1
    Accuracy 只统计正确文档排在第 1 位的题目，MRR 则对每道题使用正确文档排名的
    倒数，未命中题目贡献 0。函数不截断检索结果，确保第 K 位之后的正确文档仍会
    参与 MRR 计算。

    参数：
        golden_records: 14.1 Golden Dataset 记录，可迭代且不能为空，问题必须唯一。
        retrieved_results: 问题到完整检索结果序列的映射；每条结果必须是
            RetrievedChunk，且列表顺序必须与检索排名一致。

    返回：
        RankingReport：包含总题数、按题库顺序排列的正确文档排名、Top1 命中题数、
            0 到 1 之间的 Top1 Accuracy 和 MRR；未命中题目的排名为 None。

    异常：
        TypeError: 题库、结果映射或其中元素类型不符合接口约束时抛出。
        ValueError: 题库为空、问题重复或结果 Metadata 不是对象时抛出。
    """
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

    correct_ranks: list[int | None] = []
    reciprocal_rank_sum = 0.0
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

        correct_rank: int | None = None
        for rank, chunk in enumerate(result_list, start=1):
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("retrieved results must contain only RetrievedChunk objects")
            if not isinstance(chunk.metadata, Mapping):
                raise ValueError("RetrievedChunk.metadata must be an object")
            document_id = chunk.metadata.get("document_id") or chunk.source
            if correct_rank is None and document_id == record.expected_document:
                correct_rank = rank
        correct_ranks.append(correct_rank)
        if correct_rank is not None:
            reciprocal_rank_sum += 1 / correct_rank

    top1_hits = sum(rank == 1 for rank in correct_ranks)
    total_questions = len(records)
    return RankingReport(
        total_questions=total_questions,
        correct_ranks=tuple(correct_ranks),
        top1_hits=top1_hits,
        top1_accuracy=top1_hits / total_questions,
        mrr=reciprocal_rank_sum / total_questions,
    )
