"""阶段 14.3：演示正确文档排名、Top1 Accuracy 和 MRR 的计算。"""

import sys
from pathlib import Path


# 增加于阶段 14.3：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.golden_dataset import GoldenRecord, load_golden_dataset
from app.evaluation.ranking import evaluate_ranking
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.3：创建带 document_id 的固定检索结果，模拟指定排名。
def _chunk(document_id: str, rank: int) -> RetrievedChunk:
    """创建一条带业务文档 ID 和排名信息的检索结果。

    实现方式：使用完整 RetrievedChunk 字段保存 document_id、source 和页码，正文
    仅用于展示排名；该结果不访问向量数据库，保证 MRR / Top1 Demo 可重复运行。

    参数：
        document_id: 检索结果所属的业务文档版本 ID，必须为非空字符串。
        rank: 该结果在候选列表中的排名，必须为正整数。

    返回：
        RetrievedChunk：可传给排名评测器的固定检索结果。

    异常：
        AssertionError：Demo 传入的文档 ID 或排名不符合预期时抛出。
    """
    assert document_id
    assert rank > 0
    return RetrievedChunk(
        content=f"排名第 {rank} 的检索片段，文档为 {document_id}。",
        score=1.0 - rank / 100,
        source=document_id,
        page=1,
        metadata={
            "document_id": document_id,
            "source": document_id,
            "page": 1,
        },
    )


# 增加于阶段 14.3：构造包含第 1、3、6 位命中的固定完整检索结果。
def _build_retrieval_results(
    records: tuple[GoldenRecord, ...],
) -> dict[str, list[RetrievedChunk]]:
    """为 Golden Dataset 生成可验证不同正确文档排名的检索结果映射。

    实现方式：第一题目标文档位于第 1 位，第二题位于第 3 位，第三题位于第 6 位；
        结果映射交给正式 evaluate_ranking()，用于同时验证 Top1 和 MRR 的排名处理。

    参数：
        records: 按 14.1 题库顺序排列的 GoldenRecord 元组，必须正好包含三条记录。

    返回：
        dict[str, list[RetrievedChunk]]：问题到按排名排列的完整检索结果列表。

    异常：
        AssertionError：Demo 题库数量或记录字段不符合预期时抛出。
    """
    assert len(records) == 3
    first, second, third = records
    return {
        first.question: [
            _chunk(first.expected_document, 1),
            _chunk("other_document_a", 2),
            _chunk("other_document_b", 3),
        ],
        second.question: [
            _chunk("other_document_a", 1),
            _chunk("other_document_b", 2),
            _chunk(second.expected_document, 3),
        ],
        third.question: [
            _chunk("other_document_a", 1),
            _chunk("other_document_b", 2),
            _chunk("other_document_c", 3),
            _chunk("other_document_d", 4),
            _chunk("other_document_e", 5),
            _chunk(third.expected_document, 6),
        ],
    }


# 增加于阶段 14.3：运行排名评测并打印题目级排名和汇总指标。
def run_demo() -> None:
    """读取 Golden Dataset 并演示正确文档排名对应的 Top1 和 MRR。

    实现方式：加载 14.1 题库，构造正确文档分别位于第 1、3、6 位的完整检索结果，
    调用正式 evaluate_ranking() 计算排名指标；检查预期排名、Top1 命中数、Top1
    Accuracy 和 MRR，最后打印输入题库、关键处理结果和验收结论，不访问外部服务。

    参数：
        无入参；使用项目默认 Golden Dataset 和三条固定检索结果序列。

    返回：
        无返回值；题目级正确文档排名、Top1 Accuracy、MRR 和验收结论通过标准输出打印。

    异常：
        RuntimeError：题库数量、排名或 MRR / Top1 结果不符合预期时抛出。
    """
    records = load_golden_dataset()
    retrieval_results = _build_retrieval_results(records)
    report = evaluate_ranking(records, retrieval_results)
    expected_ranks = (1, 3, 6)
    if (
        report.total_questions != 3
        or report.correct_ranks != expected_ranks
        or report.top1_hits != 1
        or report.top1_accuracy != 1 / 3
        or report.mrr != 0.5
    ):
        raise RuntimeError("MRR / Top1 验收失败：正确文档排名或指标统计不符合预期")

    print("=== 阶段 14.3 MRR / Top1 Demo ===")
    print(f"Golden Dataset 题目数: {report.total_questions}")
    for index, rank in enumerate(report.correct_ranks, start=1):
        rank_text = str(rank) if rank is not None else "未命中"
        print(f"样本 {index} 正确文档排名: {rank_text}")
        print(f"样本 {index} Top1 命中: {'是' if rank == 1 else '否'}")
    print(f"Top1 命中题数: {report.top1_hits}")
    print(f"Top1 Accuracy: {report.top1_accuracy:.2%}")
    print(f"MRR: {report.mrr:.2%}")
    print("关键处理结果: MRR 会计入第 6 位的倒数排名，Top1 只统计第 1 位")
    print("MRR / Top1 验收: 通过")


# 增加于阶段 14.3：提供 MRR / Top1 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
