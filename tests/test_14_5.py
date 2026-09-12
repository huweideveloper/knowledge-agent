"""阶段 14.5：验证 Faithfulness 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.5：定义项目根目录，保证 Faithfulness Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FaithfulnessTest(unittest.TestCase):
    """验证生成答案是否得到正确检索资料支持。"""

    # 增加于阶段 14.5：验证 Demo 展示生成答案、证据 Metadata 和 Faithfulness 结果。
    def test_demo_evaluates_faithfulness(self) -> None:
        """运行阶段 14.5 Demo，并检查答案支持关系和平均 Faithfulness。

        实现方式：启动独立 Demo，由其读取 Golden Dataset、固定生成答案和检索证据；
        检查正确文档/页码且正文包含答案时判为支持，来源 Metadata 错误时判为不支持，
        再校验支持题数和最终 Faithfulness，确保指标来自实际输入处理。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、证据支持判断错误或指标统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_5.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.5 Faithfulness Demo", result.stdout)
        self.assertIn("Golden Dataset 题目数: 3", result.stdout)
        self.assertIn("样本 1 证据支持: 是", result.stdout)
        self.assertIn("样本 2 证据支持: 否", result.stdout)
        self.assertIn("样本 3 证据支持: 是", result.stdout)
        self.assertIn("支持答案数: 2", result.stdout)
        self.assertIn("Faithfulness: 66.67%", result.stdout)
        self.assertIn("Faithfulness 验收: 通过", result.stdout)


# 增加于阶段 14.5：提供 Faithfulness 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
