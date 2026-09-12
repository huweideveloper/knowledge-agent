"""阶段 8.4：验证 20 个问题的 Top1/Top3/Top5 检索基线。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 8.4：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RetrievalBaselineTest(unittest.TestCase):
    """验证核心问题的正确文档稳定进入 Top 5。"""

    # 增加于阶段 8.4：通过真实 Qdrant 验证 20 个问题的命中基线。
    def test_demo_records_top_one_three_five_hits(self) -> None:
        """运行阶段 8.4 Demo，并确认 20 个问题全部命中 Top 5。

        实现方式：从项目根目录启动真实 Demo，读取每个问题对应的正确
        document_id 和 Top1/Top3/Top5 命中记录，再检查问题总数为 20、每个问题
        都进入 Top5，确保基线结果可以作为后续检索评估的输入。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_8_4.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 8.4 20 问题检索基线 Demo", result.stdout)
        self.assertIn("问题总数: 20", result.stdout)
        self.assertIn("Top 5 命中: 20/20", result.stdout)
        self.assertIn("基线验收: 通过", result.stdout)
        self.assertIn("最终输出: 已记录 20 个问题的 Top1/Top3/Top5 基线", result.stdout)

        rows = re.findall(
            r"(?m)^#(\d+) .* Top1=(是|否) Top3=(是|否) Top5=(是|否)$",
            result.stdout,
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual([row[0] for row in rows], [str(index) for index in range(1, 21)])
        self.assertTrue(all(row[3] == "是" for row in rows))


# 增加于阶段 8.4：提供阶段 8.4 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
