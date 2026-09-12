"""阶段 6.3：验证 Embedding 向量维度。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.embedding.embeddings import validate_embedding_dimensions


# 增加于阶段 6.3：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmbeddingDimensionTest(unittest.TestCase):
    """验证文档向量和 Query 向量使用相同维度。"""

    # 增加于阶段 6.3：验证匹配向量返回共同维度。
    def test_matching_document_and_query_dimensions_return_dimension(self) -> None:
        """验证文档和 Query 向量维度一致时返回维度值。

        实现方式：传入两个相同维度的文档向量和一个 Query 向量，调用正式
        维度校验函数，确认校验通过并返回共同维度。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        dimension = validate_embedding_dimensions(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [0.7, 0.8, 0.9],
        )

        self.assertEqual(dimension, 3)

    # 增加于阶段 6.3：验证 Query 维度不匹配时明确拒绝。
    def test_mismatched_query_dimension_raises_value_error(self) -> None:
        """验证 Query 向量维度与文档向量不一致时抛出 ValueError。

        实现方式：传入三维文档向量和二维 Query 向量，调用正式校验函数，确认
        维度错误不会静默进入后续向量数据库流程。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：ValueError：文档和 Query 向量维度不一致时由正式函数抛出。
        """
        with self.assertRaises(ValueError):
            validate_embedding_dimensions([[0.1, 0.2, 0.3]], [0.4, 0.5])

    # 增加于阶段 6.3：验证任务级 Demo 输出真实向量维度验收结果。
    def test_demo_runs_and_reports_embedding_dimensions(self) -> None:
        """运行阶段 6.3 Demo，并检查输入、维度结果和最终输出。

        实现方式：从项目根目录启动真实 Demo，允许其复用 6.2 的 Embedding 模型
        生成文档和 Query 向量，检查进程退出码以及维度一致性的可观察输出。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_6_3.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 6.3 Embedding 维度验收 Demo", result.stdout)
        self.assertIn("文档向量维度: 512", result.stdout)
        self.assertIn("Query 向量维度: 512", result.stdout)
        self.assertIn("维度一致性: 通过", result.stdout)
        self.assertIn("最终输出: Embedding 维度验收完成", result.stdout)


# 增加于阶段 6.3：提供阶段 6.3 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
