"""阶段 8.3：验证 Top 5 检索结果按规定格式打印。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 8.3：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TopFiveOutputTest(unittest.TestCase):
    """验证终端输出包含 Top 5 的分数、来源、页码和内容。"""

    # 增加于阶段 8.3：通过真实 Qdrant 验证 Top 5 终端输出格式。
    def test_demo_prints_top_five_results(self) -> None:
        """运行阶段 8.3 Demo，并确认五条结果按文档格式打印。

        实现方式：从项目根目录启动真实 Demo，读取其标准输出，检查每条结果都
        依次包含排名分数、来源、页码和内容字段，并确认排名覆盖 1 到 5。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_8_3.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 8.3 Top 5 检索结果 Demo", result.stdout)
        self.assertIn("输入 Query: 上海住宿报销标准", result.stdout)
        self.assertIn("Top 5 验收: 通过", result.stdout)
        self.assertIn("最终输出: 终端已打印 Top 5 RetrievedChunk", result.stdout)

        ranks = re.findall(r"(?m)^#(\d+) score=([0-9.-]+)$", result.stdout)
        self.assertEqual([rank for rank, _ in ranks], ["1", "2", "3", "4", "5"])
        self.assertEqual(len(ranks), 5)

        for rank, _ in ranks:
            block_pattern = (
                rf"(?m)^#{rank} score=[0-9.-]+\n\n"
                r"来源：\n.+\n\n页码：\n.+\n\n内容：\n.+"
            )
            self.assertRegex(result.stdout, block_pattern)


# 增加于阶段 8.3：提供阶段 8.3 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
