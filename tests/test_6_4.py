"""阶段 6.4：验证 Embedding 的语义相似度实验。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

from app.embedding.embeddings import cosine_similarity


# 增加于阶段 6.4：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmbeddingSimilarityTest(unittest.TestCase):
    """验证余弦相似度计算和真实 Embedding 语义排序结果。"""

    # 增加于阶段 6.4：验证相同方向向量的余弦相似度为 1。
    def test_same_direction_vectors_have_similarity_one(self) -> None:
        """验证相同方向的向量具有最高余弦相似度。

        实现方式：使用两个同方向的简单向量调用正式相似度函数，检查返回值为
        1，证明函数实现的是标准余弦相似度而不是长度或点积比较。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [2.0, 0.0]), 1.0)

    # 增加于阶段 6.4：验证不同维度向量被拒绝。
    def test_different_dimensions_raise_value_error(self) -> None:
        """验证不同维度的向量不能计算相似度。

        实现方式：传入二维和三维向量，确认正式函数在计算前明确抛出
        ValueError，避免产生没有意义的相似度结果。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：ValueError：输入向量维度不一致时由正式函数抛出。
        """
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    # 增加于阶段 6.4：验证真实模型能区分相近语义和无关语义。
    def test_demo_reports_higher_similarity_for_a_and_b(self) -> None:
        """运行阶段 6.4 Demo，并确认 A/B 相似度高于 A/C。

        实现方式：从项目根目录启动真实 Demo，解析其打印的两个相似度数值，
        检查进程成功退出、关键输入输出存在，并验证文档规定的语义排序。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_6_4.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 6.4 语义相似度实验 Demo", result.stdout)
        self.assertIn("A：上海出差住宿上限", result.stdout)
        self.assertIn("B：上海酒店报销标准", result.stdout)
        self.assertIn("C：VPN密码如何修改", result.stdout)

        similarity_a_match = re.search(
            r"Similarity\(A,B\):\s*(-?\d+(?:\.\d+)?)", result.stdout
        )
        similarity_c_match = re.search(
            r"Similarity\(A,C\):\s*(-?\d+(?:\.\d+)?)", result.stdout
        )
        self.assertIsNotNone(similarity_a_match)
        self.assertIsNotNone(similarity_c_match)
        self.assertGreater(
            float(similarity_a_match.group(1)),
            float(similarity_c_match.group(1)),
        )
        self.assertIn("语义相似性验收: 通过", result.stdout)
        self.assertIn("最终输出: 语义相似度实验完成", result.stdout)


# 增加于阶段 6.4：提供阶段 6.4 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
