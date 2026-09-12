"""阶段 14.2：演示 Retrieval Recall@5 的计算和 Top 5 边界。"""

import sys
from pathlib import Path


# 增加于阶段 14.2：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.golden_dataset import GoldenRecord, load_golden_dataset
from app.evaluation.recall import evaluate_recall_at_k
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 14.2：创建带 document_id 的固定检索结果，模拟排名后的 Top K 输出。
def _chunk(document_id: str, rank: int) -> RetrievedChunk:
    """创建一条带业务文档 ID 和排名信息的检索结果。

    实现方式：使用完整 RetrievedChunk 字段保存 document_id、source 和页码，正文
    仅用于展示排名；该结果不访问向量数据库，保证 Recall@5 Demo 可重复运行。

    参数：
        document_id: 检索结果所属的业务文档版本 ID，必须为非空字符串。
        rank: 该结果在候选列表中的排名，必须为正整数。

    返回：
        RetrievedChunk：可传给 Recall@K 评测器的固定检索结果。

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


# 增加于阶段 14.2：构造包含第 1、5、6 位命中差异的固定检索结果。
def _build_retrieval_results(
    records: tuple[GoldenRecord, ...],
) -> dict[str, list[RetrievedChunk]]:
    """为 Golden Dataset 生成可验证 Top 5 边界的检索结果映射。

    实现方式：第一题目标文档位于第 1 位，第二题位于第 5 位，第三题位于第 6 位；
    结果映射交给正式 evaluate_recall_at_k()，因此第三题在 Recall@5 中必须未命中。

    参数：
        records: 按 14.1 题库顺序排列的 GoldenRecord 元组，必须至少包含三条记录。

    返回：
        dict[str, list[RetrievedChunk]]：问题到按排名排列的检索结果列表。

    异常：
        AssertionError：Demo 题库数量或记录字段不符合预期时抛出。
    """
    assert len(records) == 3
    first, second, third = records
    first_document = first.expected_document
    second_document = second.expected_document
    third_document = third.expected_document
    return {
        first.question: [
            _chunk(first_document, 1),
            _chunk("other_document_a", 2),
            _chunk("other_document_b", 3),
            _chunk("other_document_c", 4),
            _chunk("other_document_d", 5),
        ],
        second.question: [
            _chunk("other_document_a", 1),
            _chunk("other_document_b", 2),
            _chunk("other_document_c", 3),
            _chunk("other_document_d", 4),
            _chunk(second_document, 5),
        ],
        third.question: [
            _chunk("other_document_a", 1),
            _chunk("other_document_b", 2),
            _chunk("other_document_c", 3),
            _chunk("other_document_d", 4),
            _chunk("other_document_e", 5),
            _chunk(third_document, 6),
        ],
    }


# 增加于阶段 14.2：运行 Recall@5 评测并打印题目级命中结果和汇总指标。
def run_demo() -> None:
    """读取 Golden Dataset 并演示正确文档在 Top 5 中的命中率。

    实现方式：加载 14.1 题库，构造可重复的按排名检索结果，调用正式
    evaluate_recall_at_k() 计算 Recall@5；检查第 1、5 位命中和第 6 位不计入，最后
    打印输入题库、关键命中结果和最终召回率，不访问外部检索服务。

    参数：
        无入参；使用项目默认 Golden Dataset 和三条固定检索结果序列。

    返回：
        无返回值；题目级 Top 5 判断、汇总 Recall@5 和验收结论通过标准输出打印。

    异常：
        RuntimeError：题库数量、Top 5 命中边界或 Recall@5 结果不符合预期时抛出。
    """
    records = load_golden_dataset()
    retrieval_results = _build_retrieval_results(records)
    report = evaluate_recall_at_k(records, retrieval_results, k=5)
    expected_hits = {records[0].question, records[1].question}
    if (
        report.total_questions != 3
        or set(report.hit_questions) != expected_hits
        or report.missed_questions != (records[2].question,)
        or report.recall_at_k != 2 / 3
    ):
        raise RuntimeError("Recall@5 验收失败：Top 5 命中统计不符合预期")

    print("=== 阶段 14.2 Retrieval Recall@5 Demo ===")
    print(f"Golden Dataset 题目数: {report.total_questions}")
    print(f"Top K: {report.k}")
    for index, record in enumerate(records, start=1):
        hit = record.question in report.hit_questions
        print(f"样本 {index} 问题: {record.question}")
        print(f"  期望文档: {record.expected_document}")
        print(f"样本 {index} Top5 命中: {'是' if hit else '否'}")
    print(f"正确资料命中题数: {len(report.hit_questions)}")
    print(f"未命中题数: {len(report.missed_questions)}")
    print(f"Recall@5: {report.recall_at_k:.2%}")
    print("关键处理结果: 第 6 位正确文档不计入 Top 5")
    print("Recall@5 验收: 通过")


# 增加于阶段 14.2：提供 Recall@5 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
