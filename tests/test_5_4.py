"""阶段 5.4：验证 Chunk 抽样检查任务级 Demo。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 5.4：定义项目根目录，保证测试从任意目录执行时都能定位脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChunkInspectionDemoTest(unittest.TestCase):
    """验证阶段 5.4 Demo 能运行抽样检查并展示 Chunk 内容。"""

    # 增加于阶段 5.4：验证任务级 Demo 的输入、抽样结果和最终输出。
    def test_demo_runs_and_prints_chunk_inspection(self) -> None:
        """运行阶段 5.4 Demo，并检查随机抽样和 Chunk 内容输出。"""
        script_path = PROJECT_ROOT / "scripts" / "demo_5_4.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("输入目录", result.stdout)
        self.assertIn("原始 Document 数量", result.stdout)
        self.assertIn("Chunk 总数", result.stdout)
        self.assertIn("抽样数量: 20", result.stdout)
        self.assertIn("随机抽取", result.stdout)
        self.assertIn("chunk_id:", result.stdout)
        self.assertIn("source:", result.stdout)
        self.assertIn("上海住宿", result.stdout)
        self.assertIn("最终输出: 已完成 Chunk 抽样检查", result.stdout)


if __name__ == "__main__":
    unittest.main()
