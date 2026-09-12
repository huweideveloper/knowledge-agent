"""阶段 5.3：验证 Chunk Metadata 继承任务级 Demo。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 5.3：定义项目根目录，保证测试从任意目录执行时都能定位脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChunkMetadataDemoTest(unittest.TestCase):
    """验证阶段 5.3 Demo 能展示 Chunk 继承的业务和权限 Metadata。"""

    # 增加于阶段 5.3：验证任务级 Demo 的输入、Metadata 处理结果和最终输出。
    def test_demo_runs_and_prints_inherited_metadata(self) -> None:
        """运行阶段 5.3 Demo，并检查 Chunk 继承的关键 Metadata。"""
        script_path = PROJECT_ROOT / "scripts" / "demo_5_3.py"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("输入 Document Metadata", result.stdout)
        self.assertIn("document_id: travel_policy_2026", result.stdout)
        self.assertIn("page: 12", result.stdout)
        self.assertIn("department: finance", result.stdout)
        self.assertIn("allowed_roles: ['employee', 'manager']", result.stdout)
        self.assertIn("chunk_id:", result.stdout)
        self.assertIn("最终输出: Chunk 已继承完整 Metadata", result.stdout)


if __name__ == "__main__":
    unittest.main()
