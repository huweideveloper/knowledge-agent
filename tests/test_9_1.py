"""阶段 9.1：验证多个 RetrievedChunk 可以拼成 RAG Context。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.retrieval.vector_retriever import RetrievedChunk

from app.rag.context_builder import build_context


# 增加于阶段 9.1：定义项目根目录，保证 Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 9.1：构造仅用于 Context 格式测试的标准检索对象。
def _chunk(content: str, source: str) -> RetrievedChunk:
    """构造一个带最小来源信息的 RetrievedChunk 测试对象。

    实现方式：使用固定分数和页码创建标准结果对象，让测试只关注正文、来源和
    输入顺序，不依赖 Qdrant 或 Embedding 服务。

    参数：
        content: 测试正文，必须是字符串。
        source: 测试来源，必须是字符串。

    返回：
        RetrievedChunk：可直接传给 Context Builder 的测试结果对象。

    异常：
        无主动抛出的异常；参数类型错误时由 RetrievedChunk 数据类自然暴露。
    """
    return RetrievedChunk(
        content=content,
        score=0.9,
        source=source,
        page=1,
        metadata={"source": source, "page": 1},
    )


class ContextBuilderTest(unittest.TestCase):
    """验证 Context Builder 的固定块格式和输入顺序。"""

    # 增加于阶段 9.1：验证多个 Chunk 按需求格式拼接且顺序不变。
    def test_build_context_preserves_order_and_formats_each_chunk(self) -> None:
        """确认每个资料块包含编号、来源、正文，且顺序与输入一致。

        实现方式：传入两个可区分的 RetrievedChunk，比较完整字符串输出，覆盖
        资料编号、字段标签、块间空行和正文顺序。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = build_context(
            [_chunk("第一条正文", "制度甲"), _chunk("第二条正文", "制度乙")]
        )

        self.assertEqual(
            result,
            "[资料1]\n"
            "来源：\n制度甲\n"
            "正文：\n第一条正文\n\n"
            "[资料2]\n"
            "来源：\n制度乙\n"
            "正文：\n第二条正文",
        )

    # 增加于阶段 9.1：验证阶段 Demo 能够直接运行并打印 Context。
    def test_demo_builds_context_from_real_retrieval_results(self) -> None:
        """运行阶段 9.1 Demo，并确认输出包含输入、处理结果和 Context。

        实现方式：启动真实 Demo，由其从 Qdrant 召回两个结果并调用正式 Context
        Builder；测试只检查验收标记和格式字段，不固定向量分数或具体排名。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_9_1.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 9.1 Context Builder Demo", result.stdout)
        self.assertIn("输入 Chunk 数量: 2", result.stdout)
        self.assertIn("[资料1]", result.stdout)
        self.assertIn("来源：", result.stdout)
        self.assertIn("正文：", result.stdout)
        self.assertIn("Context Builder 验收: 通过", result.stdout)


# 增加于阶段 9.1：提供阶段自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
