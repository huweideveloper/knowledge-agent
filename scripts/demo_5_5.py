"""阶段 5.5：演示并验收 V1 Chunking 基线。"""

import sys
from pathlib import Path


# 增加于阶段 5.5：支持从项目根目录直接运行 V1 Chunking 验收 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_directory
from app.ingestion.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents,
)
from scripts.check_chunks import print_chunk_sample


# 增加于阶段 5.5：定义 V1 Chunking 验收 Demo 的输入和抽样配置。
DEFAULT_DIRECTORY = Path("data/raw")
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SAMPLE_SEED = 7


# 增加于阶段 5.5：运行 V1 Chunking 基线验收任务级 Demo。
def run_demo(directory: str | Path = DEFAULT_DIRECTORY) -> None:
    """执行阶段 5.5 V1 Chunking 验收，并打印可观察的验收证据。

    实现方式：调用正式的 load_directory() 读取原始 PDF，使用正式的
    split_documents() 按 V1 默认参数生成 Chunk，再重复切分一次比较正文和 ID
    以验证结果稳定性；随后检查 Chunk ID 唯一性、原始 Metadata 是否保留、正文
    是否为空或包含 Unicode 替换字符，最后复用阶段 5.4 的正式抽样函数打印 20
    个 Chunk 供人工检查断句、标题和政策条款完整性。Demo 不实现另一套切块逻辑。

    参数：
        directory: PDF 所在目录，可以是字符串或 pathlib.Path；默认使用 data/raw。

    返回：
        无返回值；验收输入、关键结果、抽样内容和最终状态通过标准输出打印。

    异常：
        NotADirectoryError: directory 不存在或不是目录时由 load_directory() 抛出。
        ValueError: V1 切块参数不合法时由 split_documents() 抛出。
        RuntimeError: 任一自动验收条件不满足时抛出。
    """
    directory_path = Path(directory)
    documents = load_directory(directory_path)
    chunks = split_documents(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    repeated_chunks = split_documents(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
    stable = [
        (chunk.page_content, chunk.metadata.get("chunk_id"))
        for chunk in chunks
    ] == [
        (chunk.page_content, chunk.metadata.get("chunk_id"))
        for chunk in repeated_chunks
    ]
    unique_ids = (
        bool(chunk_ids)
        and None not in chunk_ids
        and len(chunk_ids) == len(set(chunk_ids))
    )

    metadata_by_source_page = {
        (document.metadata.get("source"), document.metadata.get("page")): document.metadata
        for document in documents
    }
    metadata_complete = bool(documents) and bool(chunks)
    if metadata_complete:
        for chunk in chunks:
            expected_metadata = metadata_by_source_page.get(
                (chunk.metadata.get("source"), chunk.metadata.get("page"))
            )
            if expected_metadata is None or not (
                expected_metadata.items() <= chunk.metadata.items()
            ):
                metadata_complete = False
                break
    text_quality = bool(chunks) and all(
        chunk.page_content.strip() and "�" not in chunk.page_content
        for chunk in chunks
    )
    actual_sample_size = min(DEFAULT_SAMPLE_SIZE, len(chunks))

    print("=== 阶段 5.5 V1 Chunking 验收 Demo ===")
    print(f"输入目录: {directory_path}")
    print("切块实现: RecursiveCharacterTextSplitter")
    print(
        f"关键配置: chunk_size={DEFAULT_CHUNK_SIZE}, "
        f"chunk_overlap={DEFAULT_CHUNK_OVERLAP}"
    )
    print(f"原始 Document 数量: {len(documents)}")
    print(f"Chunk 总数: {len(chunks)}")
    print(f"稳定运行: {'通过' if stable else '失败'}")
    print(f"稳定 chunk_id: {'通过' if unique_ids else '失败'}")
    print(f"Metadata 完整继承: {'通过' if metadata_complete else '失败'}")
    print(f"基础文本质量检查: {'通过' if text_quality else '失败'}")
    print(f"抽查目标数量: {DEFAULT_SAMPLE_SIZE}")
    print(f"实际抽查数量: {actual_sample_size}")
    print("关键处理结果: 开始随机抽样，人工检查断句、标题和政策条款")
    print_chunk_sample(
        chunks,
        sample_size=DEFAULT_SAMPLE_SIZE,
        seed=DEFAULT_SAMPLE_SEED,
    )

    checks = {
        "stable": stable,
        "unique_ids": unique_ids,
        "metadata_complete": metadata_complete,
        "text_quality": text_quality,
        "sample_available": actual_sample_size > 0,
    }
    if not all(checks.values()):
        failed_checks = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V1 Chunking 验收失败：{failed_checks}")

    print("最终输出: V1 Chunking 基线验收完成")


# 增加于阶段 5.5：提供 V1 Chunking 验收 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIRECTORY)
