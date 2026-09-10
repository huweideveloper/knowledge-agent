from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from unittest import TestCase

from scripts.check_pdf_text import print_pdf_text


class PdfTextCheckTest(TestCase):
    # 增加于阶段 3.2：验证正文可读且关键制度内容存在。
    def test_prints_readable_text_from_travel_policy(self):
        output = StringIO()

        with redirect_stdout(output):
            print_pdf_text(Path("data/raw/差旅制度_2026.pdf"))

        text = output.getvalue()
        self.assertIn("上海住宿", text)
        self.assertIn("600 元/晚", text)
        self.assertNotIn("\ufffd", text)
        self.assertNotIn("\n\n\n", text)

    # 增加于阶段 3.2：验证命令行直接运行方式。
    def test_script_runs_directly_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "check_pdf_text.py"
        pdf = project_root / "data" / "raw" / "差旅制度_2026.pdf"

        result = subprocess.run(
            [sys.executable, str(script), str(pdf)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("上海住宿", result.stdout)
