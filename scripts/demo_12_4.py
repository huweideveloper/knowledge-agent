"""阶段 12.4：演示越权查询不会把受限文档带入 LLM Context。"""

import sys
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models


# 增加于阶段 12.4：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.context_builder import build_context
from app.retrieval.vector_retriever import search
from app.security import User


# 增加于阶段 12.4：执行越权检索、Context 构建和零泄露验收 Demo。
def run_demo() -> None:
    """验证 employee 用户无法把 executive 薪资文档带入 Context。

    实现方式：创建只允许 executive 角色访问的内存 Qdrant 文档，使用固定 Query
    向量后调用正式 search()；权限 Filter 在 Retrieval 阶段排除该文档，再调用
    build_context() 检查没有内容可传给 LLM，最后打印越权输入和零泄露结果。

    参数：
        无入参；使用 employee 用户和 executive_salary.pdf 受限文档示例。

    返回：
        无返回值；越权查询、过滤结果、Context 状态和验收结论通过标准输出打印。

    异常：
        RuntimeError：受限文档进入检索结果或 Context 不为空时抛出。
    """
    client = QdrantClient(":memory:")
    collection_name = "unauthorized_demo"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=collection_name,
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
            )
        ],
    )

    user = User(id="u001", department="engineering", role="employee")
    question = "高管薪资是多少？"
    with patch(
        "app.retrieval.vector_retriever.embed_query",
        return_value=[1.0, 0.0],
    ):
        results = search(
            query=question,
            top_k=5,
            client=client,
            collection_name=collection_name,
            user=user,
        )
    context = build_context(results)
    leaked_results = [
        result
        for result in results
        if result.metadata.get("document_id") == "executive_salary"
        or "500000" in result.content
    ]
    if leaked_results or context:
        raise RuntimeError("权限测试验收失败：越权数据进入了 LLM Context")

    restricted_count = client.count(collection_name=collection_name, exact=True).count
    print("=== 阶段 12.4 越权查询权限测试 Demo ===")
    print("用户: id=u001, department=engineering")
    print("用户角色: employee")
    print(f"越权问题: {question}")
    print("受限文档: executive_salary.pdf")
    print(f"受限文档数量: {restricted_count}")
    print("关键处理: Retrieval 前按 allowed_roles 过滤，越权 Chunk 不进入 Context")
    print(f"过滤后检索结果数量: {len(results)}")
    print(f"LLM Context: {'空' if not context else '非空'}")
    print(f"数据泄露数量: {len(leaked_results)}")
    print("权限测试验收: 通过")


# 增加于阶段 12.4：提供越权查询权限测试 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
