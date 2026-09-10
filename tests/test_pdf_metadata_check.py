from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import TestCase

from scripts.check_pdf_metadata import print_pdf_metadata


class PdfMetadataCheckTest(TestCase):
    # 增加于阶段 3.3：验证 source 和 page 元数据能够打印。
    def test_prints_source_and_page_metadata(self):
        output = StringIO()

        with redirect_stdout(output):
            print_pdf_metadata(Path("data/raw/差旅制度_2026.pdf"))

        text = output.getvalue()
        self.assertIn("source: data/raw/差旅制度_2026.pdf", text)
        self.assertIn("page: 0", text)
