"""阶段 14.7：验证权限越权泄露评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.7：定义项目根目录，保证权限泄露 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PermissionLeakageTest(unittest.TestCase):
    """验证越权问题不会把受限数据带入检索结果或 LLM Context。"""

    # 增加于阶段 14.7：验证 Demo 展示权限输入、泄露统计和零泄露验收结果。
    def test_demo_evaluates_zero_permission_leakage(self) -> None:
        """运行阶段 14.7 Demo，并检查越权泄露数量和泄露率均为零。

        实现方式：启动独立 Demo，由其使用内存 Qdrant、真实权限过滤检索和 Context
        Builder 处理两个 employee 越权问题；检查过滤后的结果数量、空 Context、泄露
        案例数、泄露数据数和最终泄露率，确保安全指标来自实际请求链路。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、越权数据进入结果/Context 或泄露统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_7.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.7 权限泄露评测 Demo", result.stdout)
        self.assertIn("权限评测样本数: 2", result.stdout)
        self.assertIn("用户角色: employee", result.stdout)
        self.assertIn("样本 1 过滤后检索结果数: 0", result.stdout)
        self.assertIn("样本 2 过滤后检索结果数: 0", result.stdout)
        self.assertIn("样本 1 LLM Context: 空", result.stdout)
        self.assertIn("样本 2 LLM Context: 空", result.stdout)
        self.assertIn("权限越权泄露: 0", result.stdout)
        self.assertIn("权限泄露率: 0.00%", result.stdout)
        self.assertIn("权限安全验收: 通过", result.stdout)


# 增加于阶段 14.7：提供权限泄露评测测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
