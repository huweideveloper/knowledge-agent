"""阶段 14.1：验证 Golden Dataset 可以被稳定加载和校验。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.1：定义项目根目录，保证 Golden Dataset Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GoldenDatasetTest(unittest.TestCase):
    """验证标准题库包含后续指标所需的完整 ground truth。"""

    # 增加于阶段 14.1：验证 Demo 展示题库输入、字段校验和稳定样本数量。
    def test_demo_loads_and_validates_golden_dataset(self) -> None:
        """运行阶段 14.1 Demo，并检查 Golden Dataset 的关键字段和内容。

        实现方式：启动独立 Demo 进程，由其加载项目 JSON 题库并校验每条记录，再
        检查标准输出中的文件位置、题目数量、四个 ground truth 字段、代表性题目和
        验收标记，确保题库不是只存在于源码说明中。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、字段缺失、数量变化或校验未通过时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.1 Golden Dataset Demo", result.stdout)
        self.assertIn("数据文件: data/golden_dataset.json", result.stdout)
        self.assertIn("题目数量: 3", result.stdout)
        self.assertIn(
            "字段校验: question / expected_answer / expected_document / expected_page",
            result.stdout,
        )
        self.assertIn("示例问题: 普通员工去上海出差，酒店最多报多少？", result.stdout)
        self.assertIn("示例期望文档: travel_policy_2026", result.stdout)
        self.assertIn("示例期望页码: 1", result.stdout)
        self.assertIn("Golden Dataset 验收: 通过", result.stdout)


# 增加于阶段 14.1：提供 Golden Dataset 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
