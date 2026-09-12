"""阶段 11.4：验证多轮对话中的省略和指代问题可以被补全。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.conversation import (
    ConversationHistoryStore,
    create_session_id,
    rewrite_follow_up_question,
)


# 增加于阶段 11.4：定义项目根目录，保证指代问题 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FollowUpQuestionTest(unittest.TestCase):
    """验证指代问题复用历史语义且不会凭空扩写普通问题。"""

    # 增加于阶段 11.4：验证“那经理呢？”补全为包含历史城市的完整问题。
    def test_rewrites_manager_follow_up_with_previous_city_context(self) -> None:
        """确认系统能把“那经理呢？”改写为完整的经理住宿问题。

        实现方式：使用真实 ConversationHistoryStore 保存需求文档中的上一轮 user
        问题和 assistant 回答，再把当前省略问题交给正式改写函数，检查城市、角色和
        业务主题都被正确带入输出。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 指代没有被正确补全时抛出。
        """
        store = ConversationHistoryStore()
        self.addCleanup(store.close)
        session_id = create_session_id()
        store.append_message(session_id, "user", "上海普通员工住宿标准多少？")
        store.append_message(session_id, "assistant", "普通员工住宿标准是 600 元。")

        history = store.get_messages(session_id)
        rewritten = rewrite_follow_up_question("那经理呢？", history)

        self.assertEqual(rewritten, "经理在上海出差住宿标准是多少？")

    # 增加于阶段 11.4：验证缺少可用历史时不会编造指代上下文。
    def test_keeps_question_when_history_cannot_resolve_reference(self) -> None:
        """确认普通问题或无关历史无法解析时保持原问题不变。

        实现方式：传入与住宿无关的历史消息和一个普通问题，检查改写函数不擅自
        添加城市、角色或制度信息，避免上下文不足时产生幻觉。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: 无法解析时改写了原问题时抛出。
        """
        history = []

        self.assertEqual(
            rewrite_follow_up_question("今天的会议几点？", history),
            "今天的会议几点？",
        )

    # 增加于阶段 11.4：验证 Demo 展示历史问题、指代问题和改写结果。
    def test_demo_prints_follow_up_rewrite_result(self) -> None:
        """确认 11.4 Demo 可直接运行并展示指代问题验收结果。

        实现方式：以子进程启动 Demo，读取标准输出并检查三段对话输入、关键改写
        结果和验收标记，证明 Demo 执行的是正式改写逻辑。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError: Demo 退出失败或输出缺少关键结果时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_11_4.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 11.4 指代问题 Demo", result.stdout)
        self.assertIn("历史问题: 上海普通员工住宿标准多少？", result.stdout)
        self.assertIn("当前问题: 那经理呢？", result.stdout)
        self.assertIn("改写问题: 经理在上海出差住宿标准是多少？", result.stdout)
        self.assertIn("指代问题验收: 通过", result.stdout)


# 增加于阶段 11.4：提供指代问题测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
