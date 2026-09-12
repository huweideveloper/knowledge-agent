"""阶段 10.3：验证 Citation 对答案内容的实际支持关系。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.rag.citations import (
    CitedAnswer,
    Citation,
    build_cited_answer,
    verify_citation_support,
)
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 10.3：定义项目根目录，保证 Citation 验证 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 10.3：构造包含上海住宿数字事实的检索 Chunk。
def _grounded_chunks() -> list[RetrievedChunk]:
    """返回用于支持和反例验证的真实格式检索结果。

    实现方式：构造一条明确包含 600 元/晚的 Chunk，保留 source、page 和 chunk_id，
    让测试验证内容支持而不是只验证 ID 存在。

    参数：
        无入参。

    返回：
        list[RetrievedChunk]：包含上海住宿标准事实的一条检索结果。

    异常：
        无主动抛出的异常。
    """
    return [
        RetrievedChunk(
            content="2026 年上海出差住宿标准为普通员工 600 元/晚。",
            score=0.95,
            source="travel_policy_2026",
            page=1,
            metadata={
                "source": "travel_policy_2026",
                "page": 1,
                "chunk_id": "chunk_600",
            },
            chunk_id="chunk_600",
        )
    ]


class CitationVerificationTest(unittest.TestCase):
    """验证引用 Chunk 真的包含答案中的支持事实。"""

    # 增加于阶段 10.3：验证支持答案的引用返回通过和匹配事实。
    def test_citation_passes_when_cited_chunk_contains_numeric_fact(self) -> None:
        """确认答案中的 600 元/晚确实存在于其引用 Chunk。

        实现方式：先用 10.2 映射函数生成可信 CitedAnswer，再调用内容验证函数，
        检查 supported=True 且匹配事实被记录。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        chunks = _grounded_chunks()
        cited_answer = build_cited_answer(
            {
                "answer": "普通员工去上海出差的酒店上限为 600 元/晚。",
                "citations": ["chunk_600"],
            },
            chunks,
        )

        result = verify_citation_support(cited_answer, chunks)

        self.assertTrue(result.supported)
        self.assertEqual(result.missing_facts, [])
        self.assertIn("600元/晚", result.matched_facts)

    # 增加于阶段 10.3：验证引用 Chunk 不支持答案数字时被判定为失败。
    def test_citation_fails_when_cited_chunk_does_not_contain_answer_fact(self) -> None:
        """确认把 600 元/晚的资料引用到 800 元/晚答案会失败。

        实现方式：构造一个仅改变答案金额的 CitedAnswer，调用验证函数并检查
        supported=False 和缺失事实列表，防止 Citation 只凭 ID 通过验收。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        cited_answer = CitedAnswer(
            answer="普通员工去上海出差的酒店上限为 800 元/晚。",
            citations=[
                Citation(
                    chunk_id="chunk_600",
                    source="travel_policy_2026",
                    page=1,
                    label="《travel_policy_2026》P1",
                )
            ],
        )

        result = verify_citation_support(cited_answer, _grounded_chunks())

        self.assertFalse(result.supported)
        self.assertIn("800元/晚", result.missing_facts)

    # 增加于阶段 10.3：验证 Demo 同时展示支持和不支持的判定。
    def test_demo_verifies_citation_content(self) -> None:
        """运行阶段 10.3 Demo，并确认内容验证验收通过。

        实现方式：启动真实检索 Demo，读取支持答案和篡改数字反例的验证结果，检查
        两种判定都符合预期，证明验证逻辑不是只返回固定成功值。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_10_3.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 10.3 Citation 内容验证 Demo", result.stdout)
        self.assertIn("支持答案验证: 通过", result.stdout)
        self.assertIn("错误答案验证: 拒绝", result.stdout)
        self.assertIn("Citation 内容验证验收: 通过", result.stdout)


# 增加于阶段 10.3：提供 Citation 内容验证测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
