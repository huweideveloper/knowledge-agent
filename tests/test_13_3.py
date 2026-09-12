"""阶段 13.3：验证无答案问题的 False Answer Rate 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 13.3：定义项目根目录，保证 False Answer Rate Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FalseAnswerRateTest(unittest.TestCase):
    """验证评测 Demo 能量化无答案问题是否被错误回答。"""

    # 增加于阶段 13.3：验证 Demo 展示评测样本、拒答统计和 False Answer Rate。
    def test_demo_runs_false_answer_rate_evaluation(self) -> None:
        """运行阶段 13.3 Demo，并检查无答案问题均被正确拒答。

        实现方式：启动独立 Demo 进程，让其使用真实 RagPipeline 和本地 grounded
        fallback 处理固定评测样本，再检查样本总数、无答案数量、错误回答数、FAR
        和最终验收结论，确保指标来自实际输出而不是源码文本。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、统计错误或评测结论不通过时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_13_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 13.3 False Answer Rate Demo", result.stdout)
        self.assertIn("评测样本数: 3", result.stdout)
        self.assertIn("无答案样本数: 2", result.stdout)
        self.assertIn("正确拒答数: 2", result.stdout)
        self.assertIn("错误回答数: 0", result.stdout)
        self.assertIn("False Answer Rate: 0.00%", result.stdout)
        self.assertIn("有答案样本回答: ", result.stdout)
        self.assertIn("600 元/晚", result.stdout)
        self.assertIn("False Answer Rate 验收: 通过", result.stdout)


# 增加于阶段 13.3：提供 False Answer Rate 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
