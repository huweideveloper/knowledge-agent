"""阶段 9.5：验证第一次端到端 RAG 问答结果来自检索资料。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.rag.llm import GroundedFallbackChatModel
from app.rag.pipeline import RagPipeline
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 9.5：定义项目根目录和需求文档规定的问题。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION = "普通员工去上海出差，酒店最多报多少？"


# 增加于阶段 9.5：提供包含正确住宿标准的确定性检索夹具。
def _grounded_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """返回包含上海普通员工住宿上限的检索 Chunk。

    实现方式：校验 Pipeline 传入需求问题和候选数量后，返回一条与真实 Qdrant
    payload 结构一致的标准结果对象，用于隔离端到端断言与外部服务状态。

    参数：
        query: Pipeline 传入的用户问题，必须等于需求文档问题。
        top_k: Pipeline 请求的候选数量，必须为正整数。

    返回：
        list[RetrievedChunk]：包含“普通员工 600 元/晚”证据的一条结果。

    异常：
        AssertionError: Pipeline 传入的问题或数量不符合预期时抛出。
    """
    assert query == QUESTION
    assert top_k == 5
    return [
        RetrievedChunk(
            content="2026 年上海出差住宿标准为普通员工 600 元/晚。",
            score=0.95,
            source="travel_policy_2026",
            page=1,
            metadata={"source": "travel_policy_2026", "page": 1},
        )
    ]


class FirstEndToEndTest(unittest.TestCase):
    """验证端到端答案包含检索资料中的明确数字。"""

    # 增加于阶段 9.5：验证答案和 Context 同时包含资料中的 600 元/晚。
    def test_first_end_to_end_answer_uses_retrieved_shanghai_limit(self) -> None:
        """确认模型回答没有脱离检索证据自行生成住宿标准。

        实现方式：使用固定 Retriever 和 grounded fallback 执行完整 Pipeline，分别
        检查 Context 与回答中的金额，确保答案可以追溯到本次检索资料。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = RagPipeline(
            retriever=_grounded_retriever,
            llm=GroundedFallbackChatModel(),
        ).invoke(QUESTION)

        self.assertIn("600 元/晚", result.context)
        self.assertIn("600 元/晚", result.answer)
        self.assertIn("根据提供资料", result.answer)

    # 增加于阶段 9.5：验证真实 Qdrant Demo 输出正确金额和验收结果。
    def test_demo_answers_documented_question(self) -> None:
        """运行阶段 9.5 Demo，并确认真实检索链路给出正确住宿标准。

        实现方式：启动 Demo，由其调用本地 Qdrant；在无 DEEPSEEK_API_KEY 时使用
        明确标注的 grounded fallback，在配置密钥时使用 DeepSeek，二者都必须输出
        检索资料中的 600 元/晚。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_9_5.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 9.5 第一次端到端测试 Demo", result.stdout)
        self.assertIn(QUESTION, result.stdout)
        self.assertIn("600 元/晚", result.stdout)
        self.assertIn("答案来自检索 Context: 是", result.stdout)
        self.assertIn("端到端验收: 通过", result.stdout)


# 增加于阶段 9.5：提供第一次端到端测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
