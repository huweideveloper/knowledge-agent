"""阶段 9.3：验证企业知识库问答 System Prompt。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.rag.prompt import SYSTEM_PROMPT, build_chat_prompt


# 增加于阶段 9.3：定义项目根目录，保证 Prompt Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SystemPromptTest(unittest.TestCase):
    """验证回答边界规则和 Chat Prompt 消息结构。"""

    # 增加于阶段 9.3：验证 System Prompt 包含需求文档规定的五条约束。
    def test_system_prompt_contains_grounding_rules(self) -> None:
        """确认 Prompt 禁止脱离资料回答或虚构来源。

        实现方式：逐条检查固定 System Prompt 中的中文规则，避免后续修改误删
        资料约束、未知回答和数字来源要求。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        required_rules = (
            "只能根据提供资料回答",
            "没有答案就说不知道",
            "禁止根据一般常识补充公司内部政策",
            "涉及数字必须来自资料",
            "不要虚构来源",
        )
        for rule in required_rules:
            self.assertIn(rule, SYSTEM_PROMPT)

    # 增加于阶段 9.3：验证问题和 Context 被放进不同消息角色。
    def test_build_chat_prompt_contains_question_and_context(self) -> None:
        """确认渲染结果同时包含 System、Human 消息和实际输入。

        实现方式：调用正式 Prompt 构造函数，将一个问题和资料传入，再读取
        LangChain ChatPromptValue 的消息列表，检查角色和正文。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        prompt_value = build_chat_prompt("上海住宿标准？", "普通员工 600 元/晚。")
        messages = prompt_value.to_messages()

        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertIn("只能根据提供资料回答", messages[0].content)
        self.assertIn("上海住宿标准？", messages[1].content)
        self.assertIn("普通员工 600 元/晚。", messages[1].content)

    # 增加于阶段 9.3：验证空 Prompt 输入不会生成无效请求。
    def test_build_chat_prompt_rejects_empty_question_or_context(self) -> None:
        """确认问题和资料都必须是非空字符串。

        实现方式：分别传入空问题、空资料和非字符串值，检查接口抛出明确的
        TypeError 或 ValueError，避免空 Context 被误送给后续模型。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with self.assertRaises(ValueError):
            build_chat_prompt(" ", "资料")
        with self.assertRaises(ValueError):
            build_chat_prompt("问题", " ")
        with self.assertRaises(TypeError):
            build_chat_prompt(123, "资料")  # type: ignore[arg-type]

    # 增加于阶段 9.3：验证阶段 Demo 打印固定规则和消息结构。
    def test_demo_prints_prompt_rules_and_messages(self) -> None:
        """运行阶段 9.3 Demo，并确认开发者无需阅读源码即可检查 Prompt。

        实现方式：启动无外部模型依赖的 Demo，读取 System Prompt 和 Human 消息
        的标准输出，检查输入、关键处理结果与验收标记。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_9_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 9.3 System Prompt Demo", result.stdout)
        self.assertIn("只能根据提供资料回答", result.stdout)
        self.assertIn("消息角色: system", result.stdout)
        self.assertIn("消息角色: human", result.stdout)
        self.assertIn("System Prompt 验收: 通过", result.stdout)


# 增加于阶段 9.3：提供 System Prompt 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
