import random
import sys
from collections.abc import Sequence
from pathlib import Path

# 增加于阶段 5.4：支持从项目根目录直接运行 Chunk 检查脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from app.ingestion.loader import load_directory
from app.ingestion.splitter import split_documents


# 增加于阶段 5.4：定义 Chunk 检查脚本的默认输入目录和抽样数量。
DEFAULT_DIRECTORY = Path("data/raw")
DEFAULT_SAMPLE_SIZE = 20


# 增加于阶段 5.4：随机抽样并打印 Chunk 内容供人工检查。
def print_chunk_sample(
    chunks: Sequence[Document],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> None:
    """随机抽取并打印 Chunk 的 ID、来源和正文内容。

    实现方式：使用独立的随机数生成器从 Chunk 序列中抽样；当 Chunk 数量少于
    请求数量时打印全部 Chunk。每个样本输出 Chunk ID、来源 Metadata 和正文，
    方便人工检查断句、标题保留和政策条款完整性。seed 仅用于测试时复现结果。

    参数：
        chunks: 待检查的 Chunk Document 序列。
        sample_size: 抽样数量，默认 20；超过实际数量时自动降为全部数量。
        seed: 可选随机种子；传入后结果可复现，命令行默认不传入。

    返回：
        无返回值；抽样结果通过标准输出打印。

    异常：
        ValueError: sample_size 小于或等于 0 时抛出。
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    chunk_list = list(chunks)
    selected_size = min(sample_size, len(chunk_list))
    selected_chunks = random.Random(seed).sample(chunk_list, selected_size)

    print(f"共 {len(chunk_list)} 个 Chunk，随机抽取 {selected_size} 个。")
    for sample_number, chunk in enumerate(selected_chunks, start=1):
        chunk_id = chunk.metadata.get("chunk_id", "<未生成>")
        source = chunk.metadata.get("source", "<未知>")
        print(f"--- Chunk {sample_number}/{selected_size} ---")
        print(f"chunk_id: {chunk_id}")
        print(f"source: {source}")
        print(chunk.page_content)


# 增加于阶段 5.4：提供从目录加载并检查 Chunk 的命令行入口。
if __name__ == "__main__":
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIRECTORY
    print_chunk_sample(split_documents(load_directory(directory)))
