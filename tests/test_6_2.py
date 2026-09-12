"""阶段 6.2：验证 Embedding 接口封装。"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.embedding.embeddings import embed_documents, embed_query


# 增加于阶段 6.2：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingModel:
    """为接口单测提供不访问网络的 Embedding 模型替身。"""

    # 增加于阶段 6.2：提供文档向量替身结果。
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为每个输入文本返回一个固定格式的测试向量。

        实现方式：按输入文本数量返回可区分的二维向量，不执行真实模型推理，
        仅用于验证正式封装是否正确委托给 Embeddings 实例。

        参数：
            texts: 待向量化的文本列表。

        返回：
            与输入数量一致的二维浮点向量列表。

        异常：
            无主动抛出的异常。
        """
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    # 增加于阶段 6.2：提供 Query 向量替身结果。
    def embed_query(self, query: str) -> list[float]:
        """为 Query 返回一个固定格式的测试向量。

        实现方式：返回固定二维向量，验证正式封装调用的是 Query 接口而不是
        文档批量接口。

        参数：
            query: 待向量化的用户 Query。

        返回：
            二维浮点向量。

        异常：
            无主动抛出的异常。
        """
        return [2.0, float(len(query))]


class EmbeddingInterfaceTest(unittest.TestCase):
    """验证阶段 6.2 对文档和 Query Embedding 接口的封装。"""

    # 增加于阶段 6.2：验证文档列表通过正式接口委托给模型。
    def test_embed_documents_returns_one_vector_per_text(self) -> None:
        """验证 embed_documents 返回与输入数量一致的向量列表。

        实现方式：注入测试用 Embedding 模型替身，调用正式 embed_documents()
        接口，检查输入文本被完整传递且每个文本对应一个向量。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        fake_model = FakeEmbeddingModel()

        with patch(
            "app.embedding.embeddings._get_embedding_model",
            return_value=fake_model,
        ):
            vectors = embed_documents(["上海住宿标准", "VPN 密码修改"])

        self.assertEqual(vectors, [[0.0, 1.0], [1.0, 1.0]])

    # 增加于阶段 6.2：验证用户 Query 通过独立 Query 接口向量化。
    def test_embed_query_returns_one_vector(self) -> None:
        """验证 embed_query 返回单个 Query 向量。

        实现方式：注入测试用 Embedding 模型替身，调用正式 embed_query() 接口，
        检查 Query 被传入模型并返回一维浮点向量。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        fake_model = FakeEmbeddingModel()

        with patch(
            "app.embedding.embeddings._get_embedding_model",
            return_value=fake_model,
        ):
            vectors = embed_query("上海住宿上限")

        self.assertEqual(vectors, [2.0, 6.0])

    # 增加于阶段 6.2：验证任务级 Demo 执行真实 Embedding 接口并输出向量结果。
    def test_demo_runs_and_reports_embedding_vectors(self) -> None:
        """运行阶段 6.2 Demo，并检查输入、向量输出和最终结果。

        实现方式：从项目根目录启动真实 Demo，允许其加载阶段 6.1 选定的模型，
        检查进程退出码、输入文本、文档向量和 Query 向量的可观察输出。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_6_2.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 6.2 Embedding 接口 Demo", result.stdout)
        self.assertIn("输入文本数量: 2", result.stdout)
        self.assertIn("文档向量数量: 2", result.stdout)
        self.assertIn("Query 向量已生成", result.stdout)
        self.assertIn("最终输出: Embedding 接口封装完成", result.stdout)


if __name__ == "__main__":
    unittest.main()
