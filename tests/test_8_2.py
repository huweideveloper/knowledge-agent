"""阶段 8.2：验证检索结果统一返回 RetrievedChunk 标准对象。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 8.2：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RetrievedChunkTest(unittest.TestCase):
    """验证检索结果包含正文、分数、来源、页码和 Metadata。"""

    # 增加于阶段 8.2：通过真实 Qdrant 验证标准检索对象可以被完整打印。
    def test_demo_returns_standard_retrieved_chunk_objects(self) -> None:
        """运行阶段 8.2 Demo，并确认输出包含标准对象的关键字段。

        实现方式：从项目根目录启动真实 Demo，Demo 使用现有向量 Collection 执行
        语义检索，再检查输出中的 RetrievedChunk 类型、正文、分数、来源、页码和
        Metadata 字段，确保返回结果不是只有正文字符串。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_8_2.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 8.2 标准检索对象 Demo", result.stdout)
        self.assertIn("输入 Query: 上海住宿报销标准", result.stdout)
        self.assertIn("标准对象类型: RetrievedChunk", result.stdout)
        self.assertIn("content=", result.stdout)
        self.assertIn("score=", result.stdout)
        self.assertIn("source=", result.stdout)
        self.assertIn("page=", result.stdout)
        self.assertIn("metadata=", result.stdout)
        self.assertIn("标准对象验收: 通过", result.stdout)

        count_match = re.search(r"返回结果数量: (\d+)", result.stdout)
        self.assertIsNotNone(count_match)
        self.assertGreater(int(count_match.group(1)), 0)


# 增加于阶段 8.2：提供阶段 8.2 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
