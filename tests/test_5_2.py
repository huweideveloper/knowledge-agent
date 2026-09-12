"""阶段 5.2：验证 Chunk ID 任务级 Demo。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 5.2：定义项目根目录，保证测试从任意目录执行时都能定位脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChunkIdDemoTest(unittest.TestCase):
    """验证阶段 5.2 Demo 能独立运行并展示每个 Chunk 的唯一 ID。"""

    # 增加于阶段 5.2：验证任务级 Demo 的输入、ID 生成结果和最终输出。
    def test_demo_runs_and_prints_chunk_ids(self) -> None:
        """运行阶段 5.2 Demo，并检查 Chunk ID 的关键输出。"""
        script_path = PROJECT_ROOT / "scripts" / "demo_5_2.py"
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
        self.assertIn("Chunk ID", result.stdout)
        self.assertIn("差旅制度_2026_p1_chunk_01", result.stdout)
        self.assertIn("差旅制度_2026_p1_chunk_02", result.stdout)
        self.assertIn("唯一 ID 数量=2", result.stdout)
        self.assertIn("最终输出", result.stdout)


if __name__ == "__main__":
    unittest.main()
