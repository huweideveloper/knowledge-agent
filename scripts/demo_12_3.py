"""阶段 12.3：演示 User 角色在 Qdrant Retrieval 前过滤越权文档。"""

import sys
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models


# 增加于阶段 12.3：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_retriever import search
from app.security import User
from app.security.permissions import build_permission_filter


# 增加于阶段 12.3：执行 Qdrant 内存检索和权限过滤结果展示 Demo。
def run_demo() -> None:
    """使用 Qdrant 内存 Collection 验证用户只能检索允许角色的文档。

    实现方式：创建包含员工可读文档和财务专属文档的内存 Collection，使用固定
    Query 向量避免加载 Embedding 模型，再调用正式 search()；Qdrant 在向量检索
    前应用由 User.role 生成的 Filter，最后打印过滤前后数量和实际返回文档。

    参数：
        无入参；使用 employee 用户和两条受控 Metadata 示例。

    返回：
        无返回值；权限过滤输入、关键处理结果和验收结论通过标准输出打印。

    异常：
        RuntimeError：过滤后结果包含财务文档或没有返回员工可读文档时抛出。
    """
    client = QdrantClient(":memory:")
    collection_name = "permission_demo"
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
                    "text": "员工可读的差旅制度",
                    "chunk_id": "allowed_chunk",
                    "metadata": {
                        "document_id": "travel_policy_2026",
                        "source_file": "差旅制度_2026.pdf",
                        "page": 1,
                        "allowed_roles": ["employee", "manager"],
                    },
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0],
                payload={
                    "text": "仅财务可读的内部制度",
                    "chunk_id": "secret_chunk",
                    "metadata": {
                        "document_id": "finance_secret",
                        "source_file": "财务内部制度.pdf",
                        "page": 1,
                        "allowed_roles": ["finance"],
                    },
                },
            ),
        ],
    )

    user = User(id="u001", department="engineering", role="employee")
    permission_filter = build_permission_filter(user)
    with patch(
        "app.retrieval.vector_retriever.embed_query",
        return_value=[1.0, 0.0],
    ):
        results = search(
            query="差旅制度",
            top_k=5,
            client=client,
            collection_name=collection_name,
            user=user,
        )

    returned_document_ids = [result.metadata["document_id"] for result in results]
    if returned_document_ids != ["travel_policy_2026"]:
        raise RuntimeError("权限过滤验收失败：返回了越权文档或缺少允许文档")

    candidate_count = client.count(collection_name=collection_name, exact=True).count
    print("=== 阶段 12.3 Retrieval 权限过滤 Demo ===")
    print("输入用户: id=u001, department=engineering, role=employee")
    print("用户角色: employee")
    print(
        "过滤条件: "
        f"{permission_filter.must[0].key} == {permission_filter.must[0].match.value}"
    )
    print(f"过滤前候选数量: {candidate_count}")
    print("关键处理: Qdrant 在向量检索前应用 Metadata Filter")
    print(f"过滤后结果数量: {len(results)}")
    print(f"返回文档: {', '.join(returned_document_ids)}")
    print("权限过滤验收: 通过")


# 增加于阶段 12.3：提供 Retrieval 权限过滤 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
