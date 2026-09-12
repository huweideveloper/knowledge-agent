"""阶段 14.2：验证 Retrieval Recall@5 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.2：定义项目根目录，保证 Recall@5 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecallAtFiveTest(unittest.TestCase):
    """验证正确文档是否出现在 Top 5 可以被量化。"""

    # 增加于阶段 14.2：验证 Demo 展示 Top 5 命中情况和 Recall@5 结果。
    def test_demo_evaluates_recall_at_five(self) -> None:
        """运行阶段 14.2 Demo，并检查 Top 5 边界和最终召回率。

        实现方式：启动独立 Demo，由其读取 Golden Dataset 并处理固定检索结果；检查
        目标文档在第 1、5、6 位时的命中差异、命中题数、总题数和 Recall@5，确保
        指标由实际 Top 5 结果计算而不是固定文本。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、Top 5 边界判断错误或召回率统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.2 Retrieval Recall@5 Demo", result.stdout)
        self.assertIn("Golden Dataset 题目数: 3", result.stdout)
        self.assertIn("Top K: 5", result.stdout)
        self.assertIn("样本 1 Top5 命中: 是", result.stdout)
        self.assertIn("样本 2 Top5 命中: 是", result.stdout)
        self.assertIn("样本 3 Top5 命中: 否", result.stdout)
        self.assertIn("正确资料命中题数: 2", result.stdout)
        self.assertIn("Recall@5: 66.67%", result.stdout)
        self.assertIn("Recall@5 验收: 通过", result.stdout)


# 增加于阶段 14.2：提供 Recall@5 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
