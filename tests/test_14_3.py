"""阶段 14.3：验证 MRR 和 Top1 Accuracy 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.3：定义项目根目录，保证 MRR / Top1 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RankingMetricsTest(unittest.TestCase):
    """验证正确文档排名可以被转换为 Top1 和 MRR 指标。"""

    # 增加于阶段 14.3：验证 Demo 展示逐题排名、Top1 命中和 MRR 结果。
    def test_demo_evaluates_mrr_and_top1(self) -> None:
        """运行阶段 14.3 Demo，并检查正确文档排名和两个汇总指标。

        实现方式：启动独立 Demo，由其读取 Golden Dataset 并处理固定的完整检索结果；
        检查正确文档位于第 1、3、6 位时的排名、Top1 命中差异、Top1 Accuracy 和
        MRR，确保 MRR 会统计 Top5 之外的倒数排名。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、排名判断错误或指标统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.3 MRR / Top1 Demo", result.stdout)
        self.assertIn("Golden Dataset 题目数: 3", result.stdout)
        self.assertIn("样本 1 正确文档排名: 1", result.stdout)
        self.assertIn("样本 1 Top1 命中: 是", result.stdout)
        self.assertIn("样本 2 正确文档排名: 3", result.stdout)
        self.assertIn("样本 2 Top1 命中: 否", result.stdout)
        self.assertIn("样本 3 正确文档排名: 6", result.stdout)
        self.assertIn("样本 3 Top1 命中: 否", result.stdout)
        self.assertIn("Top1 命中题数: 1", result.stdout)
        self.assertIn("Top1 Accuracy: 33.33%", result.stdout)
        self.assertIn("MRR: 50.00%", result.stdout)
        self.assertIn(
            "关键处理结果: MRR 会计入第 6 位的倒数排名，Top1 只统计第 1 位",
            result.stdout,
        )
        self.assertIn("MRR / Top1 验收: 通过", result.stdout)


# 增加于阶段 14.3：提供 MRR / Top1 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
