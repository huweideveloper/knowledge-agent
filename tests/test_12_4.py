"""阶段 12.4：验证越权查询不会把受限文档带入 LLM Context。"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from app.rag.context_builder import build_context
from app.retrieval.vector_retriever import search
from app.security import User


# 增加于阶段 12.4：定义项目根目录，保证权限测试 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 12.4：创建只包含高管薪资受限文档的 Qdrant 内存测试数据。
def _create_restricted_document_client() -> QdrantClient:
    """创建用于越权查询测试的内存 Qdrant Collection。

    实现方式：创建二维 Cosine Collection，写入一条仅 executive 角色可访问的
    executive_salary 文档；测试用户使用 employee 角色，因而该文档应在 Retrieval
    阶段被过滤掉，不会进入后续 Context。

    参数：
        无入参。

    返回：
        QdrantClient：已写入受限文档的内存客户端。

    异常：
        Exception：Qdrant 内存 Collection 创建或写入失败时透传底层异常。
    """
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="unauthorized_test",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name="unauthorized_test",
        points=[
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={
                    "text": "executive_salary.pdf：高管年度薪资为 500000 元。",
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
    return client


class UnauthorizedRetrievalTest(unittest.TestCase):
    """验证员工越权查询的检索结果和 LLM Context 均为零泄露。"""

    # 增加于阶段 12.4：验证员工不能检索 executive_salary 受限文档。
    def test_employee_cannot_put_executive_salary_into_context(self) -> None:
        """确认员工直接询问高管薪资时不会获得受限 Chunk 或 Context。

        实现方式：使用内存 Qdrant 写入仅允许 executive 的薪资文档，固定 Query
        向量后以 employee User 调用真实 search()，再把结果交给正式 Context Builder；
        断言检索为空且 Context 为空，证明越权数据在 Retrieval 层已被阻断。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：受限文档或其正文进入检索结果/Context 时抛出。
        """
        client = _create_restricted_document_client()
        user = User(id="u001", department="engineering", role="employee")
        with patch(
            "app.retrieval.vector_retriever.embed_query",
            return_value=[1.0, 0.0],
        ):
            results = search(
                query="高管薪资是多少？",
                top_k=5,
                client=client,
                collection_name="unauthorized_test",
                user=user,
            )

        context = build_context(results)
        self.assertEqual(results, [])
        self.assertEqual(context, "")
        self.assertNotIn("executive_salary", context)
        self.assertNotIn("500000", context)

    # 增加于阶段 12.4：验证 Demo 展示越权数据和 Context 均为零泄露。
    def test_demo_prints_zero_leakage_result(self) -> None:
        """确认 12.4 Demo 可直接运行并展示 0 泄露验收结果。

        实现方式：以子进程启动使用 Qdrant 内存客户端的 Demo，读取标准输出并检查
        员工身份、越权问题、过滤后数量、空 Context 和最终验收标记。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或输出显示发生泄露时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_12_4.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 12.4 越权查询权限测试 Demo", result.stdout)
        self.assertIn("用户角色: employee", result.stdout)
        self.assertIn("越权问题: 高管薪资是多少？", result.stdout)
        self.assertIn("受限文档数量: 1", result.stdout)
        self.assertIn("过滤后检索结果数量: 0", result.stdout)
        self.assertIn("LLM Context: 空", result.stdout)
        self.assertIn("数据泄露数量: 0", result.stdout)
        self.assertIn("权限测试验收: 通过", result.stdout)


# 增加于阶段 12.4：提供越权查询权限测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
