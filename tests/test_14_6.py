"""阶段 14.6：验证 Citation Accuracy 评测结果。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 14.6：定义项目根目录，保证 Citation Accuracy Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CitationAccuracyTest(unittest.TestCase):
    """验证生成 Citation 是否指向 Golden Dataset 规定的来源和页码。"""

    # 增加于阶段 14.6：验证 Demo 展示 Citation 输入、可信映射和准确率结果。
    def test_demo_evaluates_citation_accuracy(self) -> None:
        """运行阶段 14.6 Demo，并检查引用来源、页码和汇总准确率。

        实现方式：启动独立 Demo，由其把固定 LLM citation ID 映射为真实检索 Chunk 的
        Citation，再与 Golden Dataset 的期望文档和页码比较；检查两条正确引用和一条
        错误引用的逐题结果、总引用数和 Citation Accuracy，确保指标来自实际映射结果。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败、来源页码判断错误或指标统计错误时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_14_6.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 14.6 Citation Accuracy Demo", result.stdout)
        self.assertIn("Golden Dataset 题目数: 3", result.stdout)
        self.assertIn("样本 1 Citation: 《travel_policy_2026》P1", result.stdout)
        self.assertIn("样本 1 引用准确: 是", result.stdout)
        self.assertIn("样本 2 Citation: 《wrong_document》P1", result.stdout)
        self.assertIn("样本 2 引用准确: 否", result.stdout)
        self.assertIn("样本 3 Citation: 《safety_policy_2026》P2", result.stdout)
        self.assertIn("样本 3 引用准确: 是", result.stdout)
        self.assertIn("正确 Citation 数: 2", result.stdout)
        self.assertIn("总 Citation 数: 3", result.stdout)
        self.assertIn("Citation Accuracy: 66.67%", result.stdout)
        self.assertIn("Citation Accuracy 验收: 通过", result.stdout)


# 增加于阶段 14.6：提供 Citation Accuracy 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
