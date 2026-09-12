"""阶段 7.3：验证单个 Chunk 写入 Qdrant 并按 ID 查询。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 7.3：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QdrantSingleChunkTest(unittest.TestCase):
    """验证一个 Chunk 的 vector、text 和 metadata 完成写入与回读。"""

    # 增加于阶段 7.3：通过真实 Qdrant 验证单个 Point 的写入和 ID 查询。
    def test_demo_writes_and_reads_one_chunk_by_id(self) -> None:
        """运行阶段 7.3 Demo，并确认写入数据可以按 ID 完整查询回来。

        实现方式：从项目根目录启动真实 Demo，由 Demo 生成一个 Chunk 的
        Embedding、写入 Qdrant，再按固定 Point ID 查询 vector、正文和 metadata，
        检查进程成功退出以及关键数据字段均被回读。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_3.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 7.3 单个 Chunk 写入 Demo", result.stdout)
        self.assertIn(
            "Point ID: 00000000-0000-4000-8000-000000000073",
            result.stdout,
        )
        self.assertIn("写入状态: 成功", result.stdout)
        self.assertIn(
            "查询结果 ID: 00000000-0000-4000-8000-000000000073",
            result.stdout,
        )
        self.assertIn("正文: 上海出差住宿上限为普通员工 600 元/晚。", result.stdout)
        self.assertIn("Metadata: {'source': 'travel_policy_2026', 'page': 1}", result.stdout)
        self.assertIn("Vector 维度: 512", result.stdout)
        self.assertIn("按 ID 查询: 成功", result.stdout)
        self.assertIn("最终输出: 单个 Chunk 已写入并可按 ID 查询", result.stdout)


# 增加于阶段 7.3：提供阶段 7.3 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
