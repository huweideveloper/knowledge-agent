"""阶段 12.1：验证 User 身份模型能承载并校验权限判断所需字段。"""

import subprocess
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.security import User


# 增加于阶段 12.1：定义项目根目录，保证 User Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UserModelTest(unittest.TestCase):
    """验证 User 模型保存身份字段并拒绝不完整或不明确的输入。"""

    # 增加于阶段 12.1：验证合法用户信息能生成后续权限过滤可读取的对象。
    def test_builds_user_from_identity_fields(self) -> None:
        """确认 User 能保留 id、department 和 role 三个身份字段。

        实现方式：传入需求文档示例中的用户信息，读取 Pydantic 模型导出的字典，
        检查三个字段和值均可供后续权限规则使用。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：模型字段或字段值不符合需求时抛出。
        """
        user = User(id="u001", department="engineering", role="employee")

        self.assertEqual(
            user.model_dump(),
            {"id": "u001", "department": "engineering", "role": "employee"},
        )

    # 增加于阶段 12.1：验证身份字段为空或含未定义字段时被 Pydantic 拒绝。
    def test_rejects_blank_or_extra_identity_data(self) -> None:
        """确认 User 不接受空身份字段和未定义字段。

        实现方式：分别提交空白 id 和额外权限字段，检查 Pydantic 在身份模型边界
        抛出 ValidationError，避免后续权限逻辑依赖缺失或歧义数据。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：非法输入未被拒绝时抛出。
        """
        with self.assertRaises(ValidationError):
            User(id=" ", department="engineering", role="employee")
        with self.assertRaises(ValidationError):
            User(
                id="u001",
                department="engineering",
                role="employee",
                allowed_roles=["employee"],
            )

    # 增加于阶段 12.1：验证 Demo 展示用户输入、模型字段和验收结论。
    def test_demo_prints_user_model_result(self) -> None:
        """确认 12.1 Demo 可直接运行并展示 User 模型的关键结果。

        实现方式：以子进程启动 Demo，读取标准输出并检查输入字段、模型导出结果和
        验收标记，证明 Demo 调用了正式 User 模型而不是只打印固定文本。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或输出缺少关键结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_12_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 12.1 User 模型 Demo", result.stdout)
        self.assertIn("输入: id=u001, department=engineering, role=employee", result.stdout)
        self.assertIn("模型输出: {'id': 'u001', 'department': 'engineering', 'role': 'employee'}", result.stdout)
        self.assertIn("User 模型验收: 通过", result.stdout)


# 增加于阶段 12.1：提供 User 模型测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
