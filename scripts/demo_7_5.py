"""阶段 7.5：演示重复批量入库的幂等性。"""

import sys
from pathlib import Path


# 增加于阶段 7.5：支持从项目根目录直接运行幂等入库 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.ingest import ingest_documents
from app.vectorstore.qdrant_store import count_points, get_qdrant_client, wait_for_qdrant


# 增加于阶段 7.5：定义幂等入库 Demo 的默认输入目录。
SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "raw"
METADATA_DIRECTORY = PROJECT_ROOT / "data" / "metadata"


# 增加于阶段 7.5：运行两次相同批次并验收 Point 数量不增加。
def run_demo() -> None:
    """对同一批目标文档执行两次入库并验证没有重复 Point。

    实现方式：先读取 Collection 的精确 Point 数量，再连续调用两次正式
    ingest_documents()；该函数使用 document_id 生成稳定 chunk_id，并由写入层
    映射为稳定 Point ID。Demo 比较两次运行后的总数，确认第二次运行只更新已有
    Point 而没有新增重复数据，不执行旧版本删除或检索。

    参数：
        无入参；两次运行使用项目 data/raw 和 data/metadata 中的同一批文档。

    返回：
        无返回值；幂等键、运行计数和 Point 数量通过标准输出打印。

    异常：
        RuntimeError: Qdrant 不可访问，或第二次运行导致 Point 总数增加时抛出。
        ValueError、TypeError: 入库输入或 Collection 配置不符合约束时抛出。
    """
    client = get_qdrant_client()
    wait_for_qdrant(client)
    before_count = count_points(client)
    first_result = ingest_documents(
        client=client,
        source_directory=SOURCE_DIRECTORY,
        metadata_directory=METADATA_DIRECTORY,
    )
    after_first_count = count_points(client)
    second_result = ingest_documents(
        client=client,
        source_directory=SOURCE_DIRECTORY,
        metadata_directory=METADATA_DIRECTORY,
    )
    after_second_count = count_points(client)
    second_added_count = after_second_count - after_first_count

    if second_added_count != 0:
        raise RuntimeError("幂等性验收失败：第二次运行产生了重复 Point")

    print("=== 阶段 7.5 幂等入库 Demo ===")
    print(f"输入 PDF 目录: {SOURCE_DIRECTORY}")
    print(f"输入 Metadata 目录: {METADATA_DIRECTORY}")
    print("幂等依据: document_id + chunk_id")
    print(f"第一次运行写入操作数量: {first_result.points_written}")
    print(f"第二次运行写入操作数量: {second_result.points_written}")
    print(f"入库前 Point 数量: {before_count}")
    print(f"第一次运行后 Point 数量: {after_first_count}")
    print(f"第二次运行后 Point 数量: {after_second_count}")
    print(f"第二次运行新增 Point 数量: {second_added_count}")
    print("关键处理结果: 相同 chunk_id 映射到相同 Point ID，重复入库执行更新而非新增")
    print("幂等性验收: 通过")
    print("最终输出: 重复入库不会产生重复 Point")


# 增加于阶段 7.5：提供幂等入库 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
