"""阶段 5.1：验证固定长度切块任务级 Demo。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 5.1：定义项目根目录，保证测试从任意目录执行时都能定位脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedLengthChunkingDemoTest(unittest.TestCase):
    """验证阶段 5.1 Demo 能独立运行并展示固定长度切块结果。"""

    # 增加于阶段 5.1：验证任务级 Demo 的输入、处理过程和最终输出。
    def test_demo_runs_and_prints_key_results(self) -> None:
        """运行阶段 5.1 Demo，并检查关键输入、配置、结果和最终输出。"""
        script_path = PROJECT_ROOT / "scripts" / "demo_5_1.py"
        pdf_path = PROJECT_ROOT / "data" / "raw" / "差旅制度_2026.pdf"

        result = subprocess.run(
            [sys.executable, str(script_path), str(pdf_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("输入 PDF", result.stdout)
        self.assertIn("原始 Document 数量", result.stdout)
        self.assertIn("chunk_size=800", result.stdout)
        self.assertIn("chunk_overlap=150", result.stdout)
        self.assertIn("Chunk 数量", result.stdout)
        self.assertIn("上海住宿", result.stdout)
        self.assertIn("最终输出", result.stdout)


if __name__ == "__main__":
    unittest.main()
