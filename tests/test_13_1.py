"""阶段 13.1：验证最低相关度阈值可以阻止低置信度回答。"""

import subprocess
import sys
import unittest
from collections.abc import Callable
from pathlib import Path

from app.rag.pipeline import RagPipeline
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 13.1：定义项目根目录，保证无答案处理 Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 13.1：构造带固定分数的标准检索结果，隔离外部向量服务。
def _chunk(score: float) -> RetrievedChunk:
    """创建一条带指定相关度分数的测试 Chunk。

    实现方式：使用完整 RetrievedChunk 字段构造离线测试资料，仅改变 score 以覆盖
    阈值以下和阈值以上两条业务分支。

    参数：
        score: 测试用相似度分数，直接用于验证最低相关度判断。

    返回：
        RetrievedChunk：包含固定正文、来源和页码的标准检索结果。

    异常：
        无主动抛出的异常。
    """
    return RetrievedChunk(
        content="普通员工住宿标准为 600 元/晚。",
        score=score,
        source="travel_policy_2026",
        page=1,
        metadata={"source": "travel_policy_2026", "page": 1},
    )


# 增加于阶段 13.1：提供低相关度检索结果，验证 Pipeline 必须拒答。
def _low_confidence_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """返回最高分低于最低相关度阈值的检索结果。

    实现方式：校验 Pipeline 传入的 Query 和 Top K，再返回一条 score 为 0.49 的
    结果，用于证明低置信度资料不会进入 Context 或模型。

    参数：
        query: Pipeline 传入的用户问题，必须是本测试问题。
        top_k: Pipeline 请求的候选数量，必须为正整数。

    返回：
        list[RetrievedChunk]：最高分为 0.49 的单条检索结果。

    异常：
        AssertionError：Pipeline 传入的参数不符合测试预期时抛出。
    """
    assert query == "公司是否报销私人旅行？"
    assert top_k == 5
    return [_chunk(0.49)]


# 增加于阶段 13.1：提供达到阈值的检索结果，验证高相关度回答仍然正常执行。
def _high_confidence_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """返回达到最低相关度阈值的检索结果。

    实现方式：校验 Pipeline 入参并返回一条 score 为 0.5 的边界结果，确保阈值采用
    “大于等于”判断，不会误拒答刚好达到最低要求的资料。

    参数：
        query: Pipeline 传入的用户问题，必须是本测试问题。
        top_k: Pipeline 请求的候选数量，必须为正整数。

    返回：
        list[RetrievedChunk]：最高分为 0.5 的单条检索结果。

    异常：
        AssertionError：Pipeline 传入的参数不符合测试预期时抛出。
    """
    assert query == "普通员工住宿标准是多少？"
    assert top_k == 5
    return [_chunk(0.5)]


# 增加于阶段 13.1：提供可观察模型，验证拒答分支不会触发 LLM。
def _recording_model(calls: list[list[object]]) -> Callable[[list[object]], str]:
    """创建一个记录调用次数并返回固定回答的离线模型。

    实现方式：每次收到 Prompt 消息时追加到 calls，再返回固定文本，使测试能够
    区分“资料达到阈值并回答”和“资料不足而拒答”两种结果。

    参数：
        calls: 用于记录模型调用消息的可变列表。

    返回：
        callable：符合 Pipeline LLM 调用约定的测试模型。

    异常：
        无主动抛出的异常。
    """
    # 增加于阶段 13.1：实现测试模型的单次调用记录。
    def model(messages: list[object]) -> str:
        """记录一次模型调用并返回固定回答。

        实现方式：保存收到的消息列表后返回确定性文本，不访问外部模型服务。

        参数：
            messages: Pipeline 生成的 Prompt 消息列表。

        返回：
            str：固定的测试回答。

        异常：
            无主动抛出的异常。
        """
        calls.append(messages)
        return "已依据知识库资料回答。"

    return model


class RelevanceThresholdTest(unittest.TestCase):
    """验证最低相关度阈值在 RAG Pipeline 中阻止低置信度回答。"""

    # 增加于阶段 13.1：验证最高结果低于阈值时直接拒答且不调用 LLM。
    # 修改于阶段 13.2：验证阈值拒答使用统一固定话术。
    def test_low_confidence_results_abstain_before_llm(self) -> None:
        """确认低相关度检索结果不会进入 Context 或 LLM。

        实现方式：注入最高分为 0.49 的 Retriever 和可记录调用的离线模型，执行
        Pipeline 后检查返回拒答、空 Context、空 Chunk，并确认模型调用次数为零。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：低相关度结果被用于回答或模型被调用时抛出。
        """
        calls: list[list[object]] = []
        result = RagPipeline(
            retriever=_low_confidence_retriever,
            llm=_recording_model(calls),
        ).invoke("公司是否报销私人旅行？")

        self.assertEqual(result.answer, "根据当前知识库，没有找到足够信息回答该问题。")
        self.assertEqual(result.context, "")
        self.assertEqual(result.chunks, ())
        self.assertEqual(calls, [])

    # 增加于阶段 13.1：验证达到阈值的结果仍会正常进入模型回答。
    def test_threshold_boundary_keeps_relevant_results(self) -> None:
        """确认 score 等于 0.5 时资料保留并继续完成回答。

        实现方式：注入 score 为 0.5 的 Retriever 和离线模型，检查资料、Context 与
        回答均保留，覆盖最低阈值的包含边界。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：达到阈值的结果被错误过滤或未调用模型时抛出。
        """
        calls: list[list[object]] = []
        result = RagPipeline(
            retriever=_high_confidence_retriever,
            llm=_recording_model(calls),
        ).invoke("普通员工住宿标准是多少？")

        self.assertEqual(len(result.chunks), 1)
        self.assertIn("600 元/晚", result.context)
        self.assertEqual(result.answer, "已依据知识库资料回答。")
        self.assertEqual(len(calls), 1)

    # 增加于阶段 13.1：验证 Demo 展示低相关度输入、判断结果和最终拒答。
    # 修改于阶段 13.2：验证 Demo 使用统一固定拒答话术。
    def test_demo_runs_relevance_threshold_flow(self) -> None:
        """运行阶段 13.1 Demo，并检查关键输入、过滤结果和最终输出。

        实现方式：启动独立 Demo 进程，读取标准输出并验证低相关度结果被过滤且
        最终没有调用模型，确保开发者无需阅读源码即可观察本任务效果。

        参数：
            无入参。

        返回：
            无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：
            AssertionError：Demo 退出失败或输出未展示阈值拒答行为时抛出。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_13_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 13.1 最低相关度阈值 Demo", result.stdout)
        self.assertIn("输入最高相关度: 0.49", result.stdout)
        self.assertIn("最低相关度阈值: 0.50", result.stdout)
        self.assertIn("过滤后 Chunk 数量: 0", result.stdout)
        self.assertIn("模型调用次数: 0", result.stdout)
        self.assertIn(
            "最终输出: 根据当前知识库，没有找到足够信息回答该问题。",
            result.stdout,
        )
        self.assertIn("无答案处理验收: 通过", result.stdout)


# 增加于阶段 13.1：提供阈值测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
