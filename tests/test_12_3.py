"""阶段 12.3：验证 Retrieval 使用 Qdrant Filter 阻止越权文档进入结果。"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from app.security import User
from app.security.permissions import build_permission_filter
from app.retrieval.vector_retriever import search


# 增加于阶段 12.3：定义项目根目录，保证权限过滤 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RetrievalPermissionFilterTest(unittest.TestCase):
    """验证用户角色在 Qdrant Retrieval 前转换为 Metadata Filter。"""

    # 增加于阶段 12.3：验证权限过滤器匹配用户角色所在的 allowed_roles 数组。
    def test_builds_qdrant_filter_from_user_role(self) -> None:
        """确认 User 的 role 被转换为 metadata.allowed_roles 精确匹配条件。

        实现方式：构造一个 employee User，调用正式权限过滤器并读取 Qdrant Filter
        的字段条件，检查过滤路径和匹配角色均符合文档 Metadata 约定。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：过滤器路径或角色值错误时抛出。
        """
        permission_filter = build_permission_filter(
            User(id="u001", department="engineering", role="employee")
        )

        self.assertEqual(len(permission_filter.must), 1)
        condition = permission_filter.must[0]
        self.assertEqual(condition.key, "metadata.allowed_roles")
        self.assertEqual(condition.match.value, "employee")

    # 增加于阶段 12.3：验证 Qdrant 检索结果不包含用户无权访问的文档。
    def test_search_filters_out_documents_for_other_roles(self) -> None:
        """确认 employee 检索只返回 allowed_roles 包含 employee 的文档。

        实现方式：使用 Qdrant 内存客户端写入一条员工可读文档和一条仅 finance 可读
        文档，固定 Query 向量后调用真实 search()；检查返回结果只包含允许文档，
        证明 Filter 在 Qdrant 检索阶段生效而不是依赖 Prompt 隐藏结果。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：越权文档进入检索结果或允许文档未返回时抛出。
        """
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="permission_test",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name="permission_test",
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
        with patch(
            "app.retrieval.vector_retriever.embed_query",
            return_value=[1.0, 0.0],
        ):
            results = search(
                query="差旅制度",
                top_k=5,
                client=client,
                collection_name="permission_test",
                user=user,
            )

        self.assertEqual(
            [result.metadata["document_id"] for result in results],
            ["travel_policy_2026"],
        )

    # 增加于阶段 12.3：验证 Demo 展示过滤前后结果和权限验收结论。
    def test_demo_prints_permission_filtered_results(self) -> None:
        """确认 12.3 Demo 可直接运行并展示越权文档被过滤。

        实现方式：以子进程启动使用 Qdrant 内存客户端的 Demo，读取标准输出并检查
        用户角色、过滤条件、过滤前后数量和验收标记，证明 Demo 覆盖真实权限检索链路。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或输出缺少关键权限结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_12_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 12.3 Retrieval 权限过滤 Demo", result.stdout)
        self.assertIn("用户角色: employee", result.stdout)
        self.assertIn("过滤条件: metadata.allowed_roles == employee", result.stdout)
        self.assertIn("过滤前候选数量: 2", result.stdout)
        self.assertIn("过滤后结果数量: 1", result.stdout)
        self.assertIn("返回文档: travel_policy_2026", result.stdout)
        self.assertIn("权限过滤验收: 通过", result.stdout)


# 增加于阶段 12.3：提供 Retrieval 权限过滤测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
