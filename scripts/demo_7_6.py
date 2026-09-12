"""阶段 7.6：演示删除旧版本并写入新版本文档。"""

import sys
from pathlib import Path


# 增加于阶段 7.6：支持从项目根目录直接运行文档更新 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.ingest import replace_document_version
from app.vectorstore.qdrant_store import (
    count_points_by_document_id,
    get_qdrant_client,
    wait_for_qdrant,
)


# 增加于阶段 7.6：定义文档更新 Demo 的输入和版本标识。
OLD_DOCUMENT_ID = "travel_policy_2025"
NEW_DOCUMENT_ID = "travel_policy_2026"
NEW_SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "差旅制度_2026.pdf"
NEW_METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "travel_policy_2026.json"


# 增加于阶段 7.6：删除旧版本并写入新版本后验收旧 Point 不再残留。
def run_demo() -> None:
    """执行一次 2025 到 2026 差旅制度的文档版本更新并输出结果。

    实现方式：先统计旧版本 Point，再调用正式版本替换流程；流程按旧
    document_id 删除 Point，重新处理新 PDF 并写入新 document_id。Demo 最后按
    两个版本的 document_id 重新统计，确认旧版本为零且新版本有可用 Point。

    参数：
        无入参；使用项目 data/raw 和 data/metadata 中的 2026 差旅制度文件。

    返回：
        无返回值；输入、数量变化、关键字段和验收结果通过标准输出打印。

    异常：
        RuntimeError: Qdrant 不可访问，或更新后旧版本仍有 Point、新版本没有
            Point 时抛出。
        ValueError、FileNotFoundError: 更新输入或 Metadata 不符合约束时抛出。
    """
    client = get_qdrant_client()
    wait_for_qdrant(client)
    old_count_before = count_points_by_document_id(client, OLD_DOCUMENT_ID)
    update_result = replace_document_version(
        client=client,
        old_document_id=OLD_DOCUMENT_ID,
        new_source_file=NEW_SOURCE_FILE,
        new_metadata_file=NEW_METADATA_FILE,
    )
    old_count_after = count_points_by_document_id(client, OLD_DOCUMENT_ID)
    new_count_after = count_points_by_document_id(client, NEW_DOCUMENT_ID)
    deleted_count = old_count_before - old_count_after

    if old_count_after != 0:
        raise RuntimeError("文档更新验收失败：旧版本 Point 仍然存在")
    if new_count_after == 0:
        raise RuntimeError("文档更新验收失败：新版本没有写入 Point")

    print("=== 阶段 7.6 文档更新 Demo ===")
    print(f"输入旧版本 document_id: {OLD_DOCUMENT_ID}")
    print(f"输入新版本 PDF: {NEW_SOURCE_FILE}")
    print(f"输入新版本 document_id: {NEW_DOCUMENT_ID}")
    print("更新流程: 删除旧 chunks → 重新生成 → 写入新 chunks")
    print(f"更新前旧版本 Point 数量: {old_count_before}")
    print(f"删除旧版本 Point 数量: {deleted_count}")
    print(f"删除旧版本后 Point 数量: {old_count_after}")
    print(f"新版本生成 Chunk 数量: {update_result.chunks_created}")
    print(f"新版本写入 Point 数量: {update_result.points_written}")
    print(f"新版本 Point 数量: {new_count_after}")
    print(f"旧版本残留 Point 数量: {old_count_after}")
    print("关键处理结果: 旧版本按 document_id 删除，新版本按新的 document_id 写入")
    print("文档更新验收: 通过")
    print("最终输出: 更新后检索不到被废弃的旧版本内容")


# 增加于阶段 7.6：提供文档更新 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
