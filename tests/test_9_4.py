"""阶段 9.4：验证 Retriever → Context → Prompt → LLM 的 RAG Pipeline。"""

import subprocess
import sys
import unittest
from pathlib import Path

from langchain_core.messages import BaseMessage

from app.rag.pipeline import RagPipeline, RagResult
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 9.4：定义项目根目录，保证 Pipeline Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 9.4：创建 Pipeline 测试使用的固定检索结果。
def _fixture_chunks() -> list[RetrievedChunk]:
    """返回两个可重复的检索 Chunk 测试对象。

    实现方式：构造包含问题答案线索的标准 RetrievedChunk，避免单元测试依赖
    Qdrant 数据或 Embedding 模型，同时保留真实 Pipeline 所需的正文和来源字段。

    参数：
        无入参。

    返回：
        list[RetrievedChunk]：按相关度顺序排列的两个测试 Chunk。

    异常：
        无主动抛出的异常。
    """
    return [
        RetrievedChunk(
            content="普通员工去上海出差，住宿上限为 600 元/晚。",
            score=0.9,
            source="travel_policy_2026",
            page=1,
            metadata={"source": "travel_policy_2026", "page": 1},
        ),
        RetrievedChunk(
            content="出差前需要提交出差申请并获得直属经理批准。",
            score=0.8,
            source="travel_policy_2026",
            page=1,
            metadata={"source": "travel_policy_2026", "page": 1},
        ),
    ]


# 增加于阶段 9.4：提供不访问网络的测试 Retriever。
def _fake_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """返回固定检索结果并校验 Pipeline 传入的 Query 和 Top K。

    实现方式：记录接口约束并返回测试资料；函数不访问 Qdrant，用于验证 Pipeline
    的依赖注入和阶段连接顺序。

    参数：
        query: Pipeline 传入的用户问题，必须为非空字符串。
        top_k: Pipeline 请求的候选数量，必须为正整数。

    返回：
        list[RetrievedChunk]：固定的两条测试资料。

    异常：
        AssertionError: Pipeline 传入的参数不符合测试预期时抛出。
    """
    assert query == "上海住宿标准？"
    assert top_k == 5
    return _fixture_chunks()


# 增加于阶段 9.4：提供读取 Human 消息的确定性测试模型。
def _echo_model(messages: list[BaseMessage]) -> str:
    """返回 Human 消息内容，证明检索 Context 已传入模型阶段。

    实现方式：从消息列表中找到类型为 human 的消息并原样返回其正文；不调用
    网络或外部模型，只用于验证 Pipeline 的 Prompt 到 LLM 边界。

    参数：
        messages: LangChain BaseMessage 列表，必须包含一条 Human 消息。

    返回：
        str：Human 消息正文。

    异常：
        RuntimeError: 消息列表中没有 Human 消息时抛出。
    """
    for message in messages:
        if message.type == "human":
            return message.content
    raise RuntimeError("测试模型没有收到 Human 消息")


class RagPipelineTest(unittest.TestCase):
    """验证阶段 9.4 的五段式 RAG 处理链。"""

    # 增加于阶段 9.4：验证 Pipeline 将检索资料一路传递到注入模型。
    def test_pipeline_passes_retrieved_context_to_injected_model(self) -> None:
        """确认 Query、Retriever、Context Builder、Prompt 和模型顺序连通。

        实现方式：注入固定 Retriever 和 Echo 模型，执行一次 Pipeline，检查返回的
        RagResult、保留 Chunk 数量、格式化 Context 和模型回答都包含测试资料。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = RagPipeline(
            retriever=_fake_retriever,
            llm=_echo_model,
        ).invoke("上海住宿标准？")

        self.assertIsInstance(result, RagResult)
        self.assertEqual(len(result.chunks), 2)
        self.assertIn("[资料1]", result.context)
        self.assertIn("600 元/晚", result.answer)

    # 增加于阶段 9.4：验证 Pipeline 对空问题进行边界校验。
    def test_pipeline_rejects_empty_query(self) -> None:
        """确认空 Query 不会进入 Retriever 或模型调用。

        实现方式：使用固定依赖调用 Pipeline，检查空白问题在入口处抛出 ValueError。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with self.assertRaises(ValueError):
            RagPipeline(retriever=_fake_retriever, llm=_echo_model).invoke(" ")

    # 增加于阶段 9.4：验证阶段 Demo 能够展示完整处理链。
    def test_demo_runs_complete_pipeline(self) -> None:
        """运行阶段 9.4 Demo，并检查五个阶段和最终回答均有输出。

        实现方式：启动 Demo，由其使用真实 Qdrant 和默认聊天模型；无密钥环境下
        使用项目内 grounded fallback，因此测试不依赖外部网络。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_9_4.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 9.4 RAG Pipeline Demo", result.stdout)
        self.assertIn("query → retriever → context builder → prompt → LLM", result.stdout)
        self.assertIn("检索 Chunk 数量:", result.stdout)
        self.assertIn("模型回答:", result.stdout)
        self.assertIn("RAG Pipeline 验收: 通过", result.stdout)


# 增加于阶段 9.4：提供 RAG Pipeline 测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
