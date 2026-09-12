"""阶段 11.2：验证 Conversation History 保存 user 和 assistant 消息。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.conversation import (
    ConversationHistoryStore,
    ConversationMessage,
    create_session_id,
)


# 增加于阶段 11.2：定义项目根目录，保证会话历史 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConversationHistoryStoreTest(unittest.TestCase):
    """验证会话历史能保存消息、保持顺序并隔离不同会话。"""

    # 增加于阶段 11.2：验证 user 和 assistant 消息按写入顺序被保存和读取。
    def test_save_and_load_user_assistant_messages_in_order(self) -> None:
        """确认同一 session_id 下能按顺序取回 user 与 assistant 消息。

        实现方式：使用真实 SQLite 内存数据库创建存储对象，连续写入一条 user 和
        一条 assistant 消息，再比较读取结果中的角色和正文，验证历史不是只保存最后一条。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 保存或读取的消息结构、角色或顺序不符合要求时抛出。
        """
        store = ConversationHistoryStore()
        self.addCleanup(store.close)
        session_id = create_session_id()

        store.append_message(session_id, "user", "上海住宿标准是多少？")
        store.append_message(session_id, "assistant", "普通员工为 600 元/晚。")

        self.assertEqual(
            store.get_messages(session_id),
            [
                ConversationMessage("user", "上海住宿标准是多少？"),
                ConversationMessage("assistant", "普通员工为 600 元/晚。"),
            ],
        )

    # 增加于阶段 11.2：验证不同 session_id 的 Conversation History 互不泄漏。
    def test_messages_are_isolated_by_session_id(self) -> None:
        """确认不同会话读取不到彼此的消息。

        实现方式：向两个独立 session_id 分别写入不同 user 消息，再分别读取并检查
        返回结果，验证会话身份确实参与存储查询条件。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 任一会话读取到另一会话消息时抛出。
        """
        store = ConversationHistoryStore()
        self.addCleanup(store.close)
        first_session = create_session_id()
        second_session = create_session_id()

        store.append_message(first_session, "user", "第一个会话")
        store.append_message(second_session, "user", "第二个会话")

        self.assertEqual(
            store.get_messages(first_session),
            [ConversationMessage("user", "第一个会话")],
        )
        self.assertEqual(
            store.get_messages(second_session),
            [ConversationMessage("user", "第二个会话")],
        )

    # 增加于阶段 11.2：验证历史存储拒绝不支持的消息角色和空正文。
    def test_message_validation_rejects_invalid_role_and_content(self) -> None:
        """确认存储边界只接受 user/assistant 和非空消息正文。

        实现方式：分别传入 system 角色和空正文，检查正式存储接口阻断非法数据，
        防止后续 LangChain Context 混入未定义角色或空消息。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 非法输入未抛出 ValueError 时抛出。
        """
        store = ConversationHistoryStore()
        self.addCleanup(store.close)
        session_id = create_session_id()

        with self.assertRaises(ValueError):
            store.append_message(session_id, "system", "不允许")
        with self.assertRaises(ValueError):
            store.append_message(session_id, "user", " ")

    # 增加于阶段 11.2：验证存储可以显式释放 SQLAlchemy Engine 资源。
    def test_store_close_is_idempotent(self) -> None:
        """确认关闭会话历史存储不会留下连接，重复关闭也不会失败。

        实现方式：创建真实 SQLite 存储并连续调用两次 close()；该行为约束 Engine
        的连接池由存储对象负责释放，避免测试和应用退出时出现未关闭数据库警告。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError 或直接暴露异常。

        异常：
            AssertionError: 关闭后资源状态不符合预期时抛出。
            AttributeError: 存储未提供 close() 生命周期接口时由测试暴露。
        """
        store = ConversationHistoryStore()

        store.close()
        store.close()

    # 增加于阶段 11.2：验证 Demo 展示保存后的 user/assistant 历史。
    def test_demo_prints_saved_conversation_history(self) -> None:
        """确认 11.2 Demo 可直接运行并展示保存后的消息历史。

        实现方式：以子进程启动 Demo，读取标准输出并检查输入消息、读取数量、角色
        顺序和验收标记，证明 Demo 展示的是实际存储结果而非固定文本。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: Demo 退出失败或输出缺少关键结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_11_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 11.2 Conversation History Demo", result.stdout)
        self.assertIn("输入 user: 上海住宿标准是多少？", result.stdout)
        self.assertIn("输入 assistant: 普通员工为 600 元/晚。", result.stdout)
        self.assertIn("读取消息数量: 2", result.stdout)
        self.assertIn("读取角色顺序: user -> assistant", result.stdout)
        self.assertIn("Conversation History 保存验收: 通过", result.stdout)


# 增加于阶段 11.2：提供会话历史测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
