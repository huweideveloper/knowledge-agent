"""阶段 14.4：验证 Answer Correctness 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.4：定义项目根目录，保证 Answer Correctness Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnswerCorrectnessTest(unittest.TestCase):
    """验证生成答案与 Golden Dataset 期望答案可以被量化比较。"""

    # 增加于阶段 14.4：验证 Demo 展示输入答案、逐题分数和平均正确性。
    def test_demo_evaluates_answer_correctness(self) -> None:
        """运行阶段 14.4 Demo，并检查答案正确性评测结果。

        实现方式：启动独立 Demo，由其读取 Golden Dataset 并比较固定生成答案与期望
        答案；检查格式差异可被标准化、错误答案被识别、正确题数和平均 Correctness
        Score 均符合预期，确保指标来自实际输入处理而不是固定输出文本。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、逐题判断错误或正确性指标统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_4.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.4 Answer Correctness Demo", result.stdout)
        self.assertIn("Golden Dataset 题目数: 3", result.stdout)
        self.assertIn("样本 1 Correctness Score: 1.00", result.stdout)
        self.assertIn("样本 1 判定: 正确", result.stdout)
        self.assertIn("样本 2 Correctness Score: 0.00", result.stdout)
        self.assertIn("样本 2 判定: 错误", result.stdout)
        self.assertIn("样本 3 Correctness Score: 1.00", result.stdout)
        self.assertIn("样本 3 判定: 正确", result.stdout)
        self.assertIn("正确答案数: 2", result.stdout)
        self.assertIn("Answer Correctness: 66.67%", result.stdout)
        self.assertIn("Answer Correctness 验收: 通过", result.stdout)


# 增加于阶段 14.4：提供 Answer Correctness 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
