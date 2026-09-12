"""阶段 12.2：验证文档 Metadata 定义并校验 allowed_roles 权限。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.ingestion.ingest import _load_metadata_records


# 增加于阶段 12.2：定义项目根目录，保证文档权限 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentPermissionMetadataTest(unittest.TestCase):
    """验证每份文档都声明可供后续过滤使用的允许角色。"""

    # 增加于阶段 12.2：验证缺少 allowed_roles 的文档 Metadata 会被拒绝。
    def test_rejects_metadata_without_allowed_roles(self) -> None:
        """确认文档没有权限列表时不能进入后续入库流程。

        实现方式：在临时目录创建缺少 allowed_roles 的最小 Metadata JSON，调用正式
        Metadata 加载函数，检查权限字段缺失会在数据边界抛出 ValueError。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：缺少权限字段仍被接受时抛出。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "missing_permissions.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "document_id": "internal_doc",
                        "title": "内部文档",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _load_metadata_records(Path(temp_dir))

    # 增加于阶段 12.2：验证真实文档 Metadata 的允许角色可被加载并保留。
    def test_loads_real_document_allowed_roles(self) -> None:
        """确认真实差旅文档的 allowed_roles 能进入业务 Metadata。

        实现方式：读取项目 data/metadata 中的全部 JSON，按 document_id 找到 2026
        差旅制度，检查其允许角色列表与需求文档示例一致，证明权限信息没有在加载时丢失。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：文档权限列表缺失、为空或内容错误时抛出。
        """
        records = _load_metadata_records(PROJECT_ROOT / "data" / "metadata")
        travel_policy = next(
            record for record in records if record["document_id"] == "travel_policy_2026"
        )

        self.assertEqual(
            travel_policy["allowed_roles"],
            ["employee", "manager", "finance", "admin"],
        )

    # 增加于阶段 12.2：验证 Demo 打印文档权限输入、处理结果和最终输出。
    def test_demo_prints_document_permission_result(self) -> None:
        """确认 12.2 Demo 可直接运行并展示文档允许角色。

        实现方式：以子进程启动 Demo，读取标准输出并检查输入 Metadata、允许角色
        和验收标记，证明 Demo 执行的是正式 Metadata 加载逻辑。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或输出缺少关键权限结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_12_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 12.2 文档权限 Metadata Demo", result.stdout)
        self.assertIn("文档: travel_policy_2026", result.stdout)
        self.assertIn("allowed_roles: ['employee', 'manager', 'finance', 'admin']", result.stdout)
        self.assertIn("文档权限验收: 通过", result.stdout)


# 增加于阶段 12.2：提供文档权限 Metadata 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
