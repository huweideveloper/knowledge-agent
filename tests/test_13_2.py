"""阶段 13.2：验证 Prompt 规则使资料不足时输出固定拒答话术。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.rag.llm import GroundedFallbackChatModel
from app.rag.pipeline import RagPipeline
from app.rag.prompt import build_chat_prompt
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 13.2：定义项目根目录，保证 Prompt 拒答 Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 13.2：提供能进入模型但不包含问题答案的固定 Context。
def _insufficient_context_retriever(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """返回相关度达标但没有问题答案的检索结果。

    实现方式：校验 Pipeline 传入的 Query 和 Top K，返回一条 score 为 0.6 的无关
    企业资料，使阶段 13.1 允许其进入模型，再由阶段 13.2 的 Prompt 规则触发拒答。

    参数：
        query: Pipeline 传入的问题，必须是 Demo 使用的私人旅行报销问题。
        top_k: Pipeline 请求的候选数量，必须为 5。

    返回：
        list[RetrievedChunk]：包含一条不回答当前问题的标准检索结果。

    异常：
        AssertionError：Pipeline 入参不符合测试预期时抛出。
    """
    assert query == "公司是否报销私人旅行？"
    assert top_k == 5
    return [
        RetrievedChunk(
            content="员工应在每年一季度完成安全培训。",
            score=0.6,
            source="safety_policy_2026",
            page=2,
            metadata={"source": "safety_policy_2026", "page": 2},
        )
    ]


class PromptAbstentionRuleTest(unittest.TestCase):
    """验证资料不足时 Prompt 和实际 Pipeline 都使用固定拒答话术。"""

    # 增加于阶段 13.2：验证固定拒答话术被写入模型可见的 System Prompt。
    def test_system_prompt_contains_exact_abstain_answer(self) -> None:
        """确认 System Prompt 要求模型输出需求文档规定的拒答句。

        实现方式：调用正式 Prompt 构造函数，读取 System 消息并检查固定拒答句和
        “必须只回复”约束，证明规则确实会随模型请求发送。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：System Prompt 未包含固定拒答规则时抛出。
        """
        prompt_value = build_chat_prompt(
            "公司是否报销私人旅行？",
            "员工应在每年一季度完成安全培训。",
        )
        system_message = prompt_value.to_messages()[0]

        self.assertIn("必须只回复", system_message.content)
        self.assertIn(
            "根据当前知识库，没有找到足够信息回答该问题。",
            system_message.content,
        )

    # 增加于阶段 13.2：验证资料不足时本地模型适配器输出固定拒答句。
    def test_pipeline_abstains_with_exact_answer_for_insufficient_context(self) -> None:
        """确认无答案 Context 进入模型后不会生成猜测性回答。

        实现方式：注入一条 score 为 0.6 但不包含问题答案的检索结果和项目内
        GroundedFallbackChatModel，执行完整 Pipeline，检查 Context 被保留而回答
        精确等于需求文档规定的拒答话术。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：模型返回猜测、旧话术或 Context 未进入模型时抛出。
        """
        result = RagPipeline(
            retriever=_insufficient_context_retriever,
            llm=GroundedFallbackChatModel(),
        ).invoke("公司是否报销私人旅行？")

        self.assertEqual(len(result.chunks), 1)
        self.assertIn("安全培训", result.context)
        self.assertEqual(result.answer, "根据当前知识库，没有找到足够信息回答该问题。")

    # 增加于阶段 13.2：验证 Demo 展示输入、Prompt 规则和最终拒答输出。
    def test_demo_runs_prompt_abstention_flow(self) -> None:
        """运行阶段 13.2 Demo，并检查固定拒答话术确实输出。

        实现方式：启动无外部 API 依赖的 Demo，读取标准输出并验证无答案问题、
        非空但不足的 Context、Prompt 规则注入状态和最终拒答结果。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或关键拒答结果未展示时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_13_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 13.2 Prompt 拒答规则 Demo", result.stdout)
        self.assertIn("输入问题: 公司是否报销私人旅行？", result.stdout)
        self.assertIn("输入 Context: 非空但不包含问题答案", result.stdout)
        self.assertIn("Prompt 拒答规则: 已注入", result.stdout)
        self.assertIn("最终输出: 根据当前知识库，没有找到足够信息回答该问题。", result.stdout)
        self.assertIn("Prompt 拒答验收: 通过", result.stdout)


# 增加于阶段 13.2：提供 Prompt 拒答规则测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
