"""阶段 5.5：验证 V1 Chunking 基线验收。"""

import subprocess
import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.splitter import split_documents


# 增加于阶段 5.5：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V1ChunkingAcceptanceTest(unittest.TestCase):
    """验证阶段 5.5 的 V1 切块基线和任务级 Demo。"""

    # 增加于阶段 5.5：验证 chunk_size 与 chunk_overlap 可以由调用方配置。
    def test_split_documents_accepts_configurable_chunk_parameters(self) -> None:
        """验证正式切块函数接受自定义大小和重叠参数。

        实现方式：传入没有自然分隔符的长文本，使用小于默认值的切块参数，
        检查 Chunk 最大长度和相邻 Chunk 的重叠长度，确保参数确实影响正式实现。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        source_text = "0123456789" * 200
        chunks = split_documents(
            [Document(page_content=source_text, metadata={"source": "test.pdf"})],
            chunk_size=240,
            chunk_overlap=40,
        )

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk.page_content) for chunk in chunks), 240)
        self.assertEqual(chunks[1].page_content[:40], chunks[0].page_content[-40:])

    # 增加于阶段 5.5：验证任务级 Demo 输出 V1 基线的完整验收信息。
    def test_demo_runs_and_reports_v1_acceptance(self) -> None:
        """运行阶段 5.5 Demo，并检查输入、关键结果、样本和最终输出。

        实现方式：从项目根目录启动真实 Demo，读取项目原始 PDF 并检查进程退出码
        及 V1 验收所需的配置、稳定性、Metadata、ID 和 20 个样本目标输出。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_5_5.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 5.5 V1 Chunking 验收 Demo", result.stdout)
        self.assertIn("RecursiveCharacterTextSplitter", result.stdout)
        self.assertIn("chunk_size=800", result.stdout)
        self.assertIn("chunk_overlap=150", result.stdout)
        self.assertIn("稳定运行: 通过", result.stdout)
        self.assertIn("稳定 chunk_id: 通过", result.stdout)
        self.assertIn("Metadata 完整继承: 通过", result.stdout)
        self.assertIn("抽查目标数量: 20", result.stdout)
        self.assertIn("实际抽查数量:", result.stdout)
        self.assertIn("最终输出: V1 Chunking 基线验收完成", result.stdout)


if __name__ == "__main__":
    unittest.main()
