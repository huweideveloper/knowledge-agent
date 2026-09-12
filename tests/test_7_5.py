"""阶段 7.5：验证重复批量入库不会产生重复 Point。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 7.5：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IdempotentIngestionTest(unittest.TestCase):
    """验证同一批文档使用稳定业务 ID 重复入库时数量不增加。"""

    # 增加于阶段 7.5：通过真实 Qdrant 验证同一批文档重复运行的幂等性。
    def test_demo_does_not_create_duplicate_points(self) -> None:
        """运行阶段 7.5 Demo，并确认第二次入库不增加 Point 数量。

        实现方式：从项目根目录启动真实 Demo，Demo 对同一批 PDF 执行两次完整
        入库，读取两次运行后的 Collection Point 总数，检查第二次总数与第一次
        相同，并确认 Demo 输出使用 document_id/chunk_id 作为幂等依据。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_5.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 7.5 幂等入库 Demo", result.stdout)
        self.assertIn("幂等依据: document_id + chunk_id", result.stdout)
        self.assertIn("第二次运行新增 Point 数量: 0", result.stdout)
        self.assertIn("幂等性验收: 通过", result.stdout)
        self.assertIn("最终输出: 重复入库不会产生重复 Point", result.stdout)

        first_match = re.search(r"第一次运行后 Point 数量: (\d+)", result.stdout)
        second_match = re.search(r"第二次运行后 Point 数量: (\d+)", result.stdout)
        self.assertIsNotNone(first_match)
        self.assertIsNotNone(second_match)
        self.assertEqual(first_match.group(1), second_match.group(1))


# 增加于阶段 7.5：提供阶段 7.5 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
