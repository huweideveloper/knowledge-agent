"""阶段 7.6：验证文档版本更新后旧版本 chunks 被删除。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 7.6：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentUpdateTest(unittest.TestCase):
    """验证文档更新流程不会保留已废弃版本的 Point。"""

    # 增加于阶段 7.6：通过真实 Qdrant 验证旧版本 Point 在更新后归零。
    def test_demo_removes_deprecated_document_version(self) -> None:
        """运行阶段 7.6 Demo，并确认旧版本删除后没有残留 Point。

        实现方式：从项目根目录启动真实 Demo，Demo 按 document_id 删除旧版本
        chunks，再生成并写入新版本；测试检查旧版本删除后的数量、更新完成后的
        旧版本残留数量以及新版本 Point 数量。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_6.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 7.6 文档更新 Demo", result.stdout)
        self.assertIn("更新流程: 删除旧 chunks → 重新生成 → 写入新 chunks", result.stdout)
        self.assertIn("删除旧版本后 Point 数量: 0", result.stdout)
        self.assertIn("旧版本残留 Point 数量: 0", result.stdout)
        self.assertIn("文档更新验收: 通过", result.stdout)

        new_count_match = re.search(r"新版本 Point 数量: (\d+)", result.stdout)
        self.assertIsNotNone(new_count_match)
        self.assertGreater(int(new_count_match.group(1)), 0)


# 增加于阶段 7.6：提供阶段 7.6 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
