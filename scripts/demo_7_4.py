"""阶段 7.4：演示全部目标文档批量入库。"""

import sys
from pathlib import Path


# 增加于阶段 7.4：支持从项目根目录直接运行批量入库 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.ingest import ingest_documents
from app.vectorstore.qdrant_store import get_qdrant_client, wait_for_qdrant


# 增加于阶段 7.4：定义批量入库 Demo 的默认输入目录。
SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "raw"
METADATA_DIRECTORY = PROJECT_ROOT / "data" / "metadata"


# 增加于阶段 7.4：运行完整批量入库链路的任务级验收 Demo。
def run_demo() -> None:
    """执行全部目标 PDF 的批量入库并打印各阶段计数。

    实现方式：连接阶段 7.1 的 Qdrant 服务并等待其可访问，再调用正式
    ingest_documents() 完成 Loader、Cleaner、Splitter、Embedding 和 Qdrant
    流程。Demo 检查 Chunk、向量和 Point 数量一致，打印输入目录、处理阶段和
    最终验收结果；不执行检索，检索属于后续阶段。

    参数：
        无入参；输入目录固定为项目 data/raw 和 data/metadata。

    返回：
        无返回值；批量处理的关键计数和结果通过标准输出打印。

    异常：
        RuntimeError: Qdrant 不可访问、目标 PDF 未完整加载或批量结果不一致时抛出。
        ValueError、TypeError: Metadata、Embedding 或写入参数不符合约束时抛出。
    """
    client = get_qdrant_client()
    wait_for_qdrant(client)
    result = ingest_documents(
        client=client,
        source_directory=SOURCE_DIRECTORY,
        metadata_directory=METADATA_DIRECTORY,
    )

    counts_match = (
        result.chunks_created == result.vectors_created == result.points_written
    )
    if not counts_match or result.target_pdf_count == 0:
        raise RuntimeError("批量入库数量验收失败")

    print("=== 阶段 7.4 批量入库 Demo ===")
    print(f"输入 PDF 目录: {SOURCE_DIRECTORY}")
    print(f"输入 Metadata 目录: {METADATA_DIRECTORY}")
    print("处理流程: Loader → Cleaner → Splitter → Embedding → Qdrant")
    print(f"目标 PDF 数量: {result.target_pdf_count}")
    print(f"加载 Document 数量: {result.documents_loaded}")
    print(f"Chunk 数量: {result.chunks_created}")
    print(f"Embedding 向量数量: {result.vectors_created}")
    print(f"Point 写入数量: {result.points_written}")
    print("关键处理结果: 所有目标文档已完成清洗、切块、向量化和批量入库")
    print("批量入库验收: 通过")
    print("最终输出: 全部目标文档已完成批量入库")


# 增加于阶段 7.4：提供批量入库 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
