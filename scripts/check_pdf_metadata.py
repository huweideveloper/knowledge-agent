import sys
from pathlib import Path

# 增加于阶段 3.3：支持从项目根目录直接运行或以模块方式运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf


DEFAULT_PDF = Path("data/raw/差旅制度_2026.pdf")


# 增加于阶段 3.3：打印并检查按页拆分的 PDF Metadata。
def print_pdf_metadata(path: str | Path = DEFAULT_PDF) -> None:
    """打印单个 PDF 每页的 Metadata，并检查 source 和 page 是否存在。

    实现方式：调用 load_pdf() 获取按页拆分的 Document，逐页读取 metadata，
    验证 source 非空且 page 为页码整数，然后打印这两个字段供人工检查。

    参数：
        path: PDF 文件路径，可以是字符串或 pathlib.Path 对象；默认检查 2026 年差旅制度。

    返回：
        无返回值；每页的 source 和 page 通过标准输出打印。

    异常：
        ValueError: 页面缺少 source，或 page 不是整数页码时抛出。
        FileNotFoundError: path 不是一个存在的文件时由 load_pdf() 抛出。
    """
    for page_number, document in enumerate(load_pdf(path), start=1):
        source = document.metadata.get("source")
        page = document.metadata.get("page")
        if not source:
            raise ValueError(f"Page {page_number} has no source metadata")
        if not isinstance(page, int):
            raise ValueError(f"Page {page_number} has invalid page metadata: {page!r}")
        print(f"--- page {page_number} metadata ---")
        print(f"source: {source}")
        print(f"page: {page}")


# 增加于阶段 3.3：提供 Metadata 检查命令行入口。
if __name__ == "__main__":
    print_pdf_metadata(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF)
