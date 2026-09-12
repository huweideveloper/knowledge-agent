"""阶段 5.1：演示固定长度切块任务的完整执行流程。"""

import sys
from pathlib import Path


# 增加于阶段 5.1：支持从项目根目录直接运行固定长度切块 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents


# 增加于阶段 5.1：定义固定长度切块 Demo 的默认输入 PDF。
DEFAULT_PDF = Path("data/raw/差旅制度_2026.pdf")


# 增加于阶段 5.1：运行固定长度切块任务级 Demo。
def run_demo(path: str | Path = DEFAULT_PDF) -> None:
    """执行阶段 5.1 固定长度切块 Demo，并打印关键处理结果。

    实现方式：调用正式的 load_pdf() 读取 PDF，再调用正式的 split_documents()
    使用 chunk_size=800 和 chunk_overlap=150 切分 Document。脚本会依次打印
    输入 PDF、切块配置、原始 Document 数量、Chunk 数量、每个 Chunk 的长度和
    正文预览，最后打印任务完成状态；正文预览只用于展示，不会修改 Chunk 内容。

    参数：
        path: PDF 文件路径，可以是字符串或 pathlib.Path；默认使用 2026 年差旅制度。

    返回：
        无返回值；Demo 的关键输入、处理中间结果和最终输出通过标准输出打印。

    异常：
        FileNotFoundError: 输入 PDF 不存在时由 load_pdf() 抛出。
        ValueError: PDF 页码 Metadata 或切块参数不符合要求时由 split_documents() 抛出。
    """
    pdf_path = Path(path)
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    print("=== 阶段 5.1 固定长度切块 Demo ===")
    print(f"输入 PDF: {pdf_path}")
    print("关键配置: chunk_size=800, chunk_overlap=150")
    print(f"原始 Document 数量: {len(documents)}")
    print(f"Chunk 数量: {len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk.page_content.split())[:5000]
        print(f"--- Chunk {index} ---")
        print(f"长度: {len(chunk.page_content)}")
        print(f"预览: {preview}")
    print("最终输出: 已完成固定长度切块")


# 增加于阶段 5.1：提供固定长度切块 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF)
