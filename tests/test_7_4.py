"""阶段 7.4：验证全部目标文档完成批量入库。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 7.4：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BatchIngestionTest(unittest.TestCase):
    """验证 Loader 到 Qdrant 的完整批量入库链路。"""

    # 增加于阶段 7.4：通过真实 PDF、Embedding 和 Qdrant 验证批量入库。
    def test_demo_ingests_all_target_documents(self) -> None:
        """运行阶段 7.4 Demo，并确认所有目标文档均完成入库。

        实现方式：从项目根目录启动真实 Demo，读取 data/raw 下的全部目标 PDF，
        检查 Loader、Cleaner、Splitter、Embedding 和 Qdrant 各阶段的计数输出，
        并确认 Chunk、向量和 Point 数量一致且均大于零。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_4.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 7.4 批量入库 Demo", result.stdout)
        self.assertIn("处理流程: Loader → Cleaner → Splitter → Embedding → Qdrant", result.stdout)
        self.assertIn("目标 PDF 数量: 7", result.stdout)
        self.assertIn("批量入库验收: 通过", result.stdout)
        self.assertIn("最终输出: 全部目标文档已完成批量入库", result.stdout)

        chunk_match = re.search(r"Chunk 数量: (\d+)", result.stdout)
        vector_match = re.search(r"Embedding 向量数量: (\d+)", result.stdout)
        point_match = re.search(r"Point 写入数量: (\d+)", result.stdout)
        self.assertIsNotNone(chunk_match)
        self.assertIsNotNone(vector_match)
        self.assertIsNotNone(point_match)
        chunk_count = int(chunk_match.group(1))
        self.assertGreater(chunk_count, 0)
        self.assertEqual(int(vector_match.group(1)), chunk_count)
        self.assertEqual(int(point_match.group(1)), chunk_count)


# 增加于阶段 7.4：提供阶段 7.4 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
