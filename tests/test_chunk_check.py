import importlib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

from langchain_core.documents import Document


class ChunkCheckTest(TestCase):
    # 增加于阶段 5.4：验证随机抽样打印指定数量的 Chunk。
    def test_print_chunk_sample_prints_twenty_chunks_by_default(self):
        """验证 Chunk 检查函数默认随机打印 20 个 Chunk 及其内容。

        实现方式：加载真实的检查模块，传入 25 个可区分的 Document，使用
        固定随机种子复现抽样结果，并检查输出包含 20 个 Chunk 内容和 ID。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        try:
            check_module = importlib.import_module("scripts.check_chunks")
        except ModuleNotFoundError:
            check_module = None

        print_chunk_sample = getattr(check_module, "print_chunk_sample", None)
        self.assertTrue(callable(print_chunk_sample), "Chunk 检查函数尚未实现")

        chunks = [
            Document(
                page_content=f"content-{index}",
                metadata={"chunk_id": f"chunk-{index:02d}"},
            )
            for index in range(25)
        ]
        output = StringIO()

        with redirect_stdout(output):
            print_chunk_sample(chunks, seed=7)

        text = output.getvalue()
        self.assertIn("随机抽取 20 个", text)
        self.assertEqual(text.count("--- Chunk "), 20)
        self.assertEqual(text.count("content-"), 20)
        self.assertIn("chunk_id:", text)

    # 增加于阶段 5.4：验证 Chunk 检查脚本可以直接运行。
    def test_chunk_check_script_runs_from_project_root(self):
        """验证 Chunk 检查脚本能加载原始 PDF 并打印抽样结果。

        实现方式：从项目根目录启动真实脚本，读取默认 data/raw 目录，并检查
        进程退出码及已知差旅条款是否出现在标准输出中。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "check_chunks.py"

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Chunk", result.stdout)
        self.assertIn("上海住宿", result.stdout)
