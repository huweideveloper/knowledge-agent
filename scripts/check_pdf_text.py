import sys
from pathlib import Path

# 增加于阶段 3.2：支持从项目根目录直接运行或以模块方式运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf


DEFAULT_PDF = Path("data/raw/差旅制度_2026.pdf")


# 增加于阶段 3.2：打印并检查提取出的 PDF 正文。
def print_pdf_text(path: str | Path = DEFAULT_PDF) -> None:
    """打印单个 PDF 的每页正文，并检查常见的文本提取问题。

    实现方式：调用 load_pdf() 读取 PDF，逐页检查正文非空、没有乱码替换字符，
    且不存在连续大量空行；检查通过后按页号打印原始 page_content，便于人工确认。

    参数：
        path: PDF 文件路径，可以是字符串或 pathlib.Path 对象；默认检查 2026 年差旅制度。

    返回：
        无返回值；正文通过标准输出打印。

    异常：
        ValueError: 页面为空、包含乱码替换字符或存在连续大量空行时抛出。
        FileNotFoundError: path 不是一个存在的文件时由 load_pdf() 抛出。
    """
    for page_number, document in enumerate(load_pdf(path), start=1):
        content = document.page_content
        if not content.strip():
            raise ValueError(f"Page {page_number} has no text")
        if "\ufffd" in content:
            raise ValueError(f"Page {page_number} contains replacement characters")
        if "\n\n\n" in content:
            raise ValueError(f"Page {page_number} contains excessive blank lines")
        print(f"--- page {page_number} ---")
        print(content)


# 增加于阶段 3.2：提供正文检查命令行入口。
if __name__ == "__main__":
    print_pdf_text(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF)
