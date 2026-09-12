"""阶段 11.3：验证 Conversation History 暂不限制历史长度。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.conversation import ConversationHistoryStore, create_session_id


# 增加于阶段 11.3：定义项目根目录，保证历史长度 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 11.3：向真实会话存储写入可核对的多轮 user/assistant 数据。
def _append_rounds(
    store: ConversationHistoryStore,
    session_id: str,
    round_count: int,
) -> None:
    """向指定会话写入连续编号的多轮消息。

    实现方式：每轮依次写入一条 user 和一条 assistant 消息，正文使用从 0 开始的
    编号，便于测试直接判断完整历史是否保留，而不是复用被测函数计算期望值。

    参数：
        store: 已初始化的真实 ConversationHistoryStore。
        session_id: 要写入历史的会话标识，必须为非空字符串。
        round_count: 要写入的轮数，必须为非负整数。

    返回：
        无返回值；消息通过 store 持久化到对应会话。

    异常：
        ValueError: round_count 为负数时由测试辅助逻辑抛出。
        TypeError: round_count 不是整数时由测试辅助逻辑抛出。
    """
    if isinstance(round_count, bool) or not isinstance(round_count, int):
        raise TypeError("round_count must be an integer")
    if round_count < 0:
        raise ValueError("round_count must not be negative")
    for index in range(round_count):
        store.append_message(session_id, "user", f"user-{index}")
        store.append_message(session_id, "assistant", f"assistant-{index}")


class ConversationHistoryLengthTest(unittest.TestCase):
    """验证完整历史读取和 Demo 输出。"""

    # 修改于阶段 11.4：验证历史暂不限制时完整返回所有已保存轮次。
    def test_history_returns_all_rounds_without_limit(self) -> None:
        """确认 12 轮历史会完整返回，不丢弃最早消息。

        实现方式：向真实 SQLite 存储写入 12 轮消息，再读取完整 Conversation
        History，检查返回 24 条消息且从第 0 轮开始，证明当前暂不执行长度截断。

        参数：无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：AssertionError: 返回数量、首尾消息或顺序不符合要求时抛出。
        """
        store = ConversationHistoryStore()
        self.addCleanup(store.close)
        session_id = create_session_id()
        _append_rounds(store, session_id, 12)

        messages = store.get_recent_messages(session_id)

        self.assertEqual(len(messages), 24)
        self.assertEqual(messages[0].content, "user-0")
        self.assertEqual(messages[-1].content, "assistant-11")
        self.assertEqual(
            [message.role for message in messages[:2]],
            ["user", "assistant"],
        )

    # 修改于阶段 11.4：验证 Demo 直观展示未截断的完整历史。
    def test_demo_prints_unlimited_history_result(self) -> None:
        """确认 11.3 Demo 可直接运行并展示完整历史结果。

        实现方式：以子进程启动 Demo，读取标准输出并检查写入轮数、读取轮数、最早
        和最新消息以及验收标记，确保 Demo 展示历史未被截断。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: Demo 退出失败或输出缺少关键完整历史结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_11_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 11.3 Conversation History 完整历史 Demo", result.stdout)
        self.assertIn("输入历史轮数: 12", result.stdout)
        self.assertIn("读取历史轮数: 12", result.stdout)
        self.assertIn("最早读取消息: user-0", result.stdout)
        self.assertIn("最新读取消息: assistant-11", result.stdout)
        self.assertIn("完整历史读取验收: 通过", result.stdout)


# 增加于阶段 11.3：提供完整历史读取测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
