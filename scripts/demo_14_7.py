"""阶段 14.7：演示权限越权泄露评测和零泄露验收。"""

import sys
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models


# 增加于阶段 14.7：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.permission_leakage import (
    PermissionEvaluationCase,
    evaluate_permission_leakage,
)
from app.rag.context_builder import build_context
from app.retrieval.vector_retriever import search
from app.security import User


# 增加于阶段 14.7：创建包含 executive 受限文档的内存 Qdrant 数据集。
def _create_restricted_document_client() -> QdrantClient:
    """创建权限 Demo 使用的内存 Qdrant Collection。

    实现方式：创建二维 Cosine Collection，写入两条仅 executive 角色可访问的文档；
    Demo 使用 employee 用户和固定 Query 向量调用正式 search()，验证 Retrieval 前
    的 allowed_roles Filter 会阻断受限 Chunk。

    参数：
        无入参。

    返回：
        QdrantClient：已写入两条 executive 受限文档的内存客户端。

    异常：
        Exception：Qdrant 内存 Collection 创建或写入失败时抛出。
    """
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="permission_evaluation_demo",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="permission_evaluation_demo",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "text": "高管年度薪资为 500000 元。",
                    "chunk_id": "executive_salary_chunk",
                    "metadata": {
                        "document_id": "executive_salary",
                        "source_file": "executive_salary.pdf",
                        "page": 1,
                        "allowed_roles": ["executive"],
                    },
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "text": "董事会年度奖金为 900000 元。",
                    "chunk_id": "board_bonus_chunk",
                    "metadata": {
                        "document_id": "board_bonus",
                        "source_file": "board_bonus.pdf",
                        "page": 1,
                        "allowed_roles": ["executive"],
                    },
                },
            ),
        ],
    )
    return client


# 增加于阶段 14.7：运行实际权限过滤、Context 构建和泄露指标验收。
def run_demo() -> None:
    """验证 employee 越权查询不会把受限数据带入检索结果或 LLM Context。

    实现方式：创建两条仅 executive 可访问的内存 Qdrant 文档，使用 employee 用户对
    两个越权问题调用正式 search()，再调用 build_context() 保存实际输出；把用户、
    受限文档、敏感金额、检索结果和 Context 交给 evaluate_permission_leakage()，最后
    打印每个权限样本和整体泄露指标。Demo 不访问外部模型或外部 Qdrant 服务。

    参数：
        无入参；使用 employee 用户和两个 executive 受限文档样本。

    返回：
        无返回值；权限输入、过滤结果、Context 状态、泄露统计和验收结论通过标准输出打印。

    异常：
        RuntimeError：受限数据进入结果/Context，或权限泄露率不为零时抛出。
    """
    client = _create_restricted_document_client()
    user = User(id="u001", department="engineering", role="employee")
    evaluation_inputs = (
        ("高管薪资是多少？", "executive_salary", "500000"),
        ("董事会奖金是多少？", "board_bonus", "900000"),
    )
    cases: list[PermissionEvaluationCase] = []
    with patch(
        "app.retrieval.vector_retriever.embed_query",
        return_value=[1.0, 0.0],
    ):
        for question, document_id, marker in evaluation_inputs:
            results = search(
                query=question,
                top_k=5,
                client=client,
                collection_name="permission_evaluation_demo",
                user=user,
            )
            context = build_context(results)
            cases.append(
                PermissionEvaluationCase(
                    question=question,
                    user=user,
                    restricted_document_ids=frozenset({document_id}),
                    restricted_markers=frozenset({marker}),
                    retrieved_results=tuple(results),
                    context=context,
                )
            )

    report = evaluate_permission_leakage(cases)
    if (
        report.total_cases != 2
        or report.total_retrieved_chunks != 0
        or report.leaked_cases != 0
        or report.leaked_chunks != 0
        or report.leaked_contexts != 0
        or report.leakage_rate != 0.0
    ):
        raise RuntimeError("权限安全验收失败：存在越权数据泄露")

    restricted_count = client.count(
        collection_name="permission_evaluation_demo",
        exact=True,
    ).count
    print("=== 阶段 14.7 权限泄露评测 Demo ===")
    print(f"权限评测样本数: {report.total_cases}")
    print("用户: id=u001, department=engineering")
    print(f"用户角色: {user.role}")
    print(f"受限文档总数: {restricted_count}")
    print("关键处理结果: Retrieval 前按 allowed_roles 过滤，结果再交给 Context Builder")
    for index, (case, evaluation_input) in enumerate(
        zip(cases, evaluation_inputs),
        start=1,
    ):
        question, document_id, _ = evaluation_input
        leaked = question in report.leaked_case_questions
        print(f"样本 {index} 越权问题: {question}")
        print(f"  受限文档: {document_id}")
        print(f"样本 {index} 过滤后检索结果数: {len(case.retrieved_results)}")
        print(f"样本 {index} LLM Context: {'空' if not case.context else '非空'}")
        print(f"样本 {index} 数据泄露: {'是' if leaked else '否'}")
    print(f"权限越权泄露: {report.leaked_cases}")
    print(f"泄露 Chunk 数: {report.leaked_chunks}")
    print(f"泄露 Context 数: {report.leaked_contexts}")
    print(f"权限泄露率: {report.leakage_rate:.2%}")
    print("权限安全验收: 通过")


# 增加于阶段 14.7：提供权限泄露评测 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
