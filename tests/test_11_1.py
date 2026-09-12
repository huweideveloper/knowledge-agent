"""阶段 11.1：验证新会话能生成稳定格式的 session_id。"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

from app.conversation import create_session_id


# 增加于阶段 11.1：定义项目根目录，保证会话 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SessionIdTest(unittest.TestCase):
    """验证会话标识具备可识别格式且不同会话不会复用。"""

    # 增加于阶段 11.1：验证新会话返回 session_ 前缀和 UUID 十六进制标识。
    def test_create_session_id_has_expected_format(self) -> None:
        """确认新会话标识符合 session_<uuid> 格式。

        实现方式：调用正式会话标识生成函数，再用正则表达式校验前缀和 UUID
        十六进制长度，确保后续历史存储可以依赖统一键格式。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 返回值不符合需求格式时抛出。
        """
        session_id = create_session_id()

        self.assertRegex(session_id, re.compile(r"^session_[0-9a-f]{32}$"))

    # 增加于阶段 11.1：验证连续创建的会话标识彼此隔离。
    def test_create_session_id_is_unique_for_new_sessions(self) -> None:
        """确认两次创建新会话不会得到相同的 session_id。

        实现方式：连续创建两个新会话并比较返回值，验证不同用户或新会话不会
        误用同一个历史键。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 两次调用生成相同标识时抛出。
        """
        first = create_session_id()
        second = create_session_id()

        self.assertNotEqual(first, second)

    # 增加于阶段 11.1：验证 Demo 输出新会话输入、处理结果和最终 session_id。
    def test_demo_prints_session_id(self) -> None:
        """确认 11.1 Demo 可直接运行并展示会话标识。

        实现方式：以子进程启动项目 Demo，读取标准输出并校验输入说明、生成结果
        和验收标记，避免 Demo 只存在于源码而无法独立验证。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: Demo 退出失败或输出缺少关键结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_11_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 11.1 会话 session_id Demo", result.stdout)
        self.assertIn("输入: 新会话", result.stdout)
        self.assertRegex(result.stdout, r"输出 session_id: session_[0-9a-f]{32}")
        self.assertIn("session_id 生成验收: 通过", result.stdout)


# 增加于阶段 11.1：提供会话标识测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
