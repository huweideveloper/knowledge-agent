"""阶段 5.2：演示为每个 Chunk 生成唯一 ID 的完整执行流程。"""

import sys
from pathlib import Path


# 增加于阶段 5.2：支持从项目根目录直接运行 Chunk ID Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents


# 增加于阶段 5.2：定义 Chunk ID Demo 的默认输入 PDF。
DEFAULT_PDF = Path("data/raw/差旅制度_2026.pdf")


# 增加于阶段 5.2：运行 Chunk ID 任务级 Demo。
def run_demo(path: str | Path = DEFAULT_PDF) -> None:
    """执行阶段 5.2 Chunk ID Demo，并打印每个 Chunk 的可追溯标识。

    实现方式：调用正式的 load_pdf() 读取 PDF，再调用正式的 split_documents()
    完成切块和 Chunk ID 生成。脚本会打印输入 PDF、原始 Document 数量、Chunk
    数量、每个 Chunk 的 ID 及其来源页码，最后统计 ID 总数和唯一 ID 数量，帮助
    开发者确认每个 Chunk 都获得了可用于更新、删除、引用和评测的唯一标识。

    参数：
        path: PDF 文件路径，可以是字符串或 pathlib.Path；默认使用 2026 年差旅制度。

    返回：
        无返回值；Demo 的关键输入、处理结果和最终输出通过标准输出打印。

    异常：
        FileNotFoundError: 输入 PDF 不存在时由 load_pdf() 抛出。
        ValueError: PDF 页码 Metadata 或切块参数不符合要求时由 split_documents() 抛出。
    """
    pdf_path = Path(path)
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    print("=== 阶段 5.2 Chunk ID Demo ===")
    print(f"输入 PDF: {pdf_path}")
    print(f"原始 Document 数量: {len(documents)}")
    print(f"Chunk 数量: {len(chunks)}")
    print("关键处理结果: 已为每个 Chunk 生成 chunk_id")
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.metadata["chunk_id"]
        page = chunk.metadata.get("page", "<未知>")
        print(f"--- Chunk {index} ---")
        print(f"Chunk ID: {chunk_id}")
        print(f"来源页码: {page}")
    print(f"最终输出: ID 总数={len(chunk_ids)}，唯一 ID 数量={len(set(chunk_ids))}")


# 增加于阶段 5.2：提供 Chunk ID Demo 的命令行入口。
if __name__ == "__main__":
    run_demo(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF)
