"""阶段 10.2：验证结构化 LLM 输出映射为可信 Citation。"""

import subprocess
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.rag.citations import CitedAnswer, StructuredAnswer, build_cited_answer
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 10.2：定义项目根目录，保证 Citation 映射 Demo 可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 10.2：构造带唯一 Chunk ID 的 Citation 映射测试数据。
def _chunks() -> list[RetrievedChunk]:
    """返回两条包含来源、页码和 Chunk ID 的测试检索结果。

    实现方式：使用与阶段 10.1 相同的标准 RetrievedChunk 字段，确保 Citation
    映射只依赖真实检索结果已有的来源数据。

    参数：
        无入参。

    返回：
        list[RetrievedChunk]：可被结构化 citation ID 引用的两条结果。

    异常：
        无主动抛出的异常。
    """
    return [
        RetrievedChunk(
            content="上海住宿标准为普通员工 600 元/晚。",
            score=0.95,
            source="差旅管理制度.pdf",
            page=12,
            metadata={
                "source": "差旅管理制度.pdf",
                "page": 12,
                "chunk_id": "chunk_001",
            },
            chunk_id="chunk_001",
        ),
        RetrievedChunk(
            content="出差前需要提交申请。",
            score=0.8,
            source="差旅管理制度.pdf",
            page=13,
            metadata={
                "source": "差旅管理制度.pdf",
                "page": 13,
                "chunk_id": "chunk_002",
            },
            chunk_id="chunk_002",
        ),
    ]


class CitationMappingTest(unittest.TestCase):
    """验证结构化答案只能引用实际检索 Chunk。"""

    # 增加于阶段 10.2：验证合法 Chunk ID 映射为来源和页码展示。
    def test_maps_structured_citations_from_existing_chunk_map(self) -> None:
        """确认程序根据 chunk_id 生成可信来源标签，而不是采用模型页码。

        实现方式：传入 Pydantic 结构化答案和两条真实格式检索结果，只声明引用
        chunk_001，检查输出保留回答并映射为《差旅管理制度.pdf》P12。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = build_cited_answer(
            StructuredAnswer(
                answer="普通员工上海住宿上限为 600 元/晚。",
                citations=["chunk_001"],
            ),
            _chunks(),
        )

        self.assertIsInstance(result, CitedAnswer)
        self.assertEqual(result.answer, "普通员工上海住宿上限为 600 元/晚。")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].chunk_id, "chunk_001")
        self.assertEqual(result.citations[0].source, "差旅管理制度.pdf")
        self.assertEqual(result.citations[0].page, 12)
        self.assertEqual(result.citations[0].label, "《差旅管理制度.pdf》P12")

    # 增加于阶段 10.2：验证不存在的 Chunk ID 不会被映射成虚构 Citation。
    def test_rejects_citation_id_outside_chunk_map(self) -> None:
        """确认模型引用未知 ID 时抛错，而不是接受模型自行生成的来源。

        实现方式：传入 chunk_999 这个不存在于检索结果的 ID，要求映射函数抛出
        ValueError，阻断虚构文件名和页码流向用户。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with self.assertRaises(ValueError):
            build_cited_answer(
                {"answer": "无法确认。", "citations": ["chunk_999"]},
                _chunks(),
            )

    # 增加于阶段 10.2：验证 Pydantic 拒绝缺少必填结构化字段的模型输出。
    def test_structured_answer_requires_answer_and_citations(self) -> None:
        """确认 LLM 输出必须包含正文和引用 ID 列表。

        实现方式：分别省略 answer 和 citations，检查 Pydantic 在进入业务映射前
        抛出 ValidationError，避免使用不完整对象继续生成 Citation。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with self.assertRaises(ValidationError):
            StructuredAnswer.model_validate({"citations": ["chunk_001"]})
        with self.assertRaises(ValidationError):
            StructuredAnswer.model_validate({"answer": "回答"})

    # 增加于阶段 10.2：验证 Demo 展示结构化输入到可信输出的转换。
    def test_demo_maps_structured_output_to_trusted_citation(self) -> None:
        """运行阶段 10.2 Demo，并确认最终展示来源来自 Chunk Metadata。

        实现方式：启动不调用外部 LLM 的 Demo，由固定结构化答案引用真实检索结果；
        测试检查输入 ID、映射后的来源页码和验收结论。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_10_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 10.2 可信 Citation 映射 Demo", result.stdout)
        self.assertIn('LLM 结构化输出: {"answer":', result.stdout)
        self.assertIn('"citations": ["', result.stdout)
        self.assertIn("映射 Citation: 《", result.stdout)
        self.assertIn("Citation 映射验收: 通过", result.stdout)


# 增加于阶段 10.2：提供 Citation 映射测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
