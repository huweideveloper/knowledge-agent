"""阶段 10.1：验证每个检索结果保留 Citation 所需来源字段。"""

from types import SimpleNamespace
import subprocess
import sys
import unittest
from pathlib import Path

from app.retrieval.vector_retriever import RetrievedChunk, _to_retrieved_chunk


# 增加于阶段 10.1：定义项目根目录，保证 Citation 来源 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 10.1：构造带完整来源 Metadata 的 Qdrant Point 测试替身。
def _point_with_source() -> SimpleNamespace:
    """创建包含正文、来源、页码和稳定 Chunk ID 的最小 Point。

    实现方式：使用 SimpleNamespace 模拟 Qdrant ScoredPoint，仅提供正式转换函数
    读取的 payload 和 score 字段，让测试聚焦来源字段映射。

    参数：
        无入参。

    返回：
        SimpleNamespace：可传入 `_to_retrieved_chunk()` 的测试 Point。

    异常：
        无主动抛出的异常。
    """
    return SimpleNamespace(
        payload={
            "text": "普通员工上海住宿上限为 600 元/晚。",
            "metadata": {
                "source": "travel_policy_2026",
                "page": 12,
                "chunk_id": "travel_policy_2026_p12_chunk_01",
            },
        },
        score=0.95,
    )


class CitationSourceRetentionTest(unittest.TestCase):
    """验证标准检索对象保留 Citation 所需的三项来源信息。"""

    # 增加于阶段 10.1：验证 source、page、chunk_id 映射到标准结果对象。
    def test_retrieved_chunk_retains_source_page_and_chunk_id(self) -> None:
        """确认来源字段既可直接访问，也继续保存在完整 Metadata 中。

        实现方式：将带来源 Metadata 的 Point 交给正式转换函数，检查标准对象的
        source、page、chunk_id 和 metadata 四处结果，确保后续 Citation 不必猜来源。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = _to_retrieved_chunk(_point_with_source())

        self.assertIsInstance(result, RetrievedChunk)
        self.assertEqual(result.source, "travel_policy_2026")
        self.assertEqual(result.page, 12)
        self.assertEqual(result.chunk_id, "travel_policy_2026_p12_chunk_01")
        self.assertEqual(
            result.metadata["chunk_id"],
            "travel_policy_2026_p12_chunk_01",
        )

    # 增加于阶段 10.1：验证缺少 Chunk ID 的 Point 不会进入可引用结果。
    def test_retrieved_chunk_rejects_missing_chunk_id(self) -> None:
        """确认没有 chunk_id 的检索结果会在来源边界被拒绝。

        实现方式：删除测试 Point 的 chunk_id 后调用转换函数，要求抛出 RuntimeError，
        防止后续 Citation 生成不存在或不可追溯的引用。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        point = _point_with_source()
        del point.payload["metadata"]["chunk_id"]

        with self.assertRaises(RuntimeError):
            _to_retrieved_chunk(point)

    # 增加于阶段 10.1：验证真实检索 Demo 打印每条结果的来源三元组。
    def test_demo_prints_source_page_and_chunk_id(self) -> None:
        """运行阶段 10.1 Demo，并确认每条结果都有三项来源字段。

        实现方式：启动真实 Qdrant 检索 Demo，读取其结构化来源输出；测试检查验收
        标记和字段标签，不固定向量分数、排名或具体 Chunk ID。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_10_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 10.1 检索来源保留 Demo", result.stdout)
        self.assertIn("source=", result.stdout)
        self.assertIn("page=", result.stdout)
        self.assertIn("chunk_id=", result.stdout)
        self.assertIn("来源保留验收: 通过", result.stdout)


# 增加于阶段 10.1：提供 Citation 来源保留测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
