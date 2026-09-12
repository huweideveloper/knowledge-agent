"""阶段 5.4：演示随机抽样检查 Chunk 内容的完整执行流程。"""

import sys
from pathlib import Path


# 增加于阶段 5.4：支持从项目根目录直接运行 Chunk 抽样检查 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.loader import load_directory
from app.ingestion.splitter import split_documents
from scripts.check_chunks import print_chunk_sample


# 增加于阶段 5.4：定义 Chunk 抽样检查 Demo 的默认输入目录和参数。
DEFAULT_DIRECTORY = Path("data/raw")
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 7


# 增加于阶段 5.4：运行 Chunk 抽样检查任务级 Demo。
def run_demo(
    directory: str | Path = DEFAULT_DIRECTORY,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> None:
    """执行阶段 5.4 Chunk 抽样检查 Demo，并打印可供人工检查的内容。

    实现方式：调用正式的 load_directory() 读取输入目录中的 PDF，再调用正式的
    split_documents() 生成 Chunk，最后复用正式的 print_chunk_sample() 随机抽取
    最多 20 个 Chunk。脚本会打印输入目录、原始 Document 数量、Chunk 总数、
    抽样配置以及 Chunk 的 ID、来源和正文，帮助人工检查断句、标题保留和政策
    条款完整性。Demo 固定随机种子只为让示例输出可复现，不改变正式抽样逻辑。

    参数：
        directory: PDF 所在目录，可以是字符串或 pathlib.Path；默认使用 data/raw。
        sample_size: 抽样数量，必须为正整数；默认抽取 20 个，实际不足时打印全部。
        seed: 随机种子，用于复现 Demo 的抽样顺序；默认使用 7。

    返回：
        无返回值；检查输入、处理中间结果和抽样内容通过标准输出打印。

    异常：
        NotADirectoryError: directory 不存在或不是目录时由 load_directory() 抛出。
        ValueError: sample_size 小于或等于 0，或切块参数不合法时抛出。
    """
    directory_path = Path(directory)
    documents = load_directory(directory_path)
    chunks = split_documents(documents)

    print("=== 阶段 5.4 Chunk 抽样检查 Demo ===")
    print(f"输入目录: {directory_path}")
    print(f"抽样数量: {sample_size}")
    print(f"随机种子: {seed}")
    print(f"原始 Document 数量: {len(documents)}")
    print(f"Chunk 总数: {len(chunks)}")
    print("关键处理结果: 开始随机抽样检查 Chunk 的断句、标题和政策条款")
    print_chunk_sample(chunks, sample_size=sample_size, seed=seed)
    print("最终输出: 已完成 Chunk 抽样检查")


# 增加于阶段 5.4：提供 Chunk 抽样检查 Demo 的命令行入口。
if __name__ == "__main__":
    input_directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIRECTORY
    run_demo(input_directory)
