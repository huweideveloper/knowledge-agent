from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.ingestion import loader
from app.ingestion.loader import load_pdf


class PdfLoaderTest(TestCase):
    # 增加于阶段 3.1：验证单个 PDF Loader。
    def test_load_pdf_returns_documents_for_each_page(self):
        path = Path("data/raw/差旅制度_2026.pdf")

        documents = load_pdf(path)

        self.assertEqual(len(documents), 1)
        self.assertIn("上海住宿", documents[0].page_content)
        self.assertEqual(documents[0].metadata["source"], str(path))

    # 增加于阶段 3.4：验证目录 Loader 会按稳定顺序读取全部 PDF。
    def test_load_directory_returns_documents_from_all_pdf_files(self):
        load_directory = getattr(loader, "load_directory", None)
        self.assertTrue(callable(load_directory), "目录 Loader 尚未实现")

        documents = load_directory(Path("data/raw"))

        expected_names = [
            "IT服务手册.pdf",
            "VPN操作手册.pdf",
            "员工手册.pdf",
            "差旅制度_2025.pdf",
            "差旅制度_2026.pdf",
            "请假管理制度.pdf",
            "费用报销制度.pdf",
        ]
        actual_names = [Path(document.metadata["source"]).name for document in documents]

        self.assertEqual(actual_names, expected_names)
        self.assertEqual(len(documents), len(expected_names))

    # 增加于阶段 3.5：验证损坏 PDF 会被记录并跳过，不影响其他文件。
    def test_load_directory_skips_invalid_pdf_and_records_error(self):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            valid_pdf = directory / "有效文档.pdf"
            invalid_pdf = directory / "损坏文档.pdf"
            copyfile(Path("data/raw/差旅制度_2026.pdf"), valid_pdf)
            invalid_pdf.write_bytes("这不是一个有效的 PDF 文件".encode())

            caught_error = None
            documents = []
            with self.assertLogs("app.ingestion.loader", level="WARNING") as logs:
                try:
                    documents = loader.load_directory(directory)
                except Exception as error:  # noqa: BLE001 - 验证批量加载不会向上抛出坏文件异常。
                    caught_error = error

            self.assertIsNone(caught_error, "坏 PDF 不应让目录加载失败")
            self.assertEqual(len(documents), 1)
            self.assertIn("上海住宿", documents[0].page_content)
            log_output = "\n".join(logs.output)
            self.assertIn("损坏文档.pdf", log_output)
            self.assertIn("原因", log_output)
