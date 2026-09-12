"""阶段 7.3：演示写入一个 Chunk 并按 ID 查询回来。"""

import sys
from pathlib import Path


# 增加于阶段 7.3：支持从项目根目录直接运行单个 Chunk 写入 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import embed_documents
from app.ingestion.ingest import read_chunk, write_chunk
from app.vectorstore.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    get_qdrant_client,
    wait_for_qdrant,
)


# 增加于阶段 7.3：定义文档规定的单个 Chunk 输入。
# 修改于阶段 7.3：使用 Qdrant 支持的 UUID 字符串作为 Point ID。
CHUNK_ID = "00000000-0000-4000-8000-000000000073"
CHUNK_TEXT = "上海出差住宿上限为普通员工 600 元/晚。"
CHUNK_METADATA = {"source": "travel_policy_2026", "page": 1}


# 增加于阶段 7.3：运行单个 Chunk 的向量、正文和 Metadata 写入验收 Demo。
def run_demo() -> None:
    """生成一个 Chunk 的向量，写入 Qdrant 并按 ID 验证完整回读。

    实现方式：使用阶段 6 Embedding 接口生成正文向量，确认阶段 7.2 Collection
    已存在后，调用正式 write_chunk() 写入 Point，再调用 read_chunk() 按固定 ID
    查询。Demo 检查回读的 ID、正文、Metadata 和向量维度，不执行批量入库或搜索。

    参数：
        无入参；Chunk 正文、Metadata 和 ID 使用脚本内置验收样例。

    返回：
        无返回值；写入前输入、回读字段和最终结果通过标准输出打印。

    异常：
        RuntimeError: Qdrant 不可访问、Collection 不存在或回读内容不一致时抛出。
        LookupError: 写入后按 ID 查询不到 Point 时由正式查询函数抛出。
        ValueError、TypeError: Embedding 或 Chunk 输入不符合约束时由正式接口抛出。
    """
    client = get_qdrant_client()
    wait_for_qdrant(client)
    if not client.collection_exists(collection_name=DEFAULT_COLLECTION_NAME):
        raise RuntimeError("Collection 不存在，请先运行 scripts/demo_7_2.py")

    vector = embed_documents([CHUNK_TEXT])[0]
    write_chunk(
        client=client,
        chunk_id=CHUNK_ID,
        text=CHUNK_TEXT,
        vector=vector,
        metadata=CHUNK_METADATA,
    )
    record = read_chunk(client=client, chunk_id=CHUNK_ID)
    payload = record.payload or {}
    stored_vector = record.vector
    stored_text = payload.get("text")
    stored_metadata = payload.get("metadata")

    if (
        record.id != CHUNK_ID
        or stored_text != CHUNK_TEXT
        or stored_metadata != CHUNK_METADATA
        or not stored_vector
    ):
        raise RuntimeError("按 ID 查询回的数据与写入内容不一致")
    output_metadata = {
        key: stored_metadata[key]
        for key in CHUNK_METADATA
    }

    print("=== 阶段 7.3 单个 Chunk 写入 Demo ===")
    print(f"Point ID: {CHUNK_ID}")
    print(f"输入正文: {CHUNK_TEXT}")
    print(f"输入 Metadata: {CHUNK_METADATA}")
    print(f"输入 Vector 维度: {len(vector)}")
    print("写入状态: 成功")
    print(f"查询结果 ID: {record.id}")
    print(f"正文: {stored_text}")
    print(f"Metadata: {output_metadata}")
    print(f"Vector 维度: {len(stored_vector)}")
    print("按 ID 查询: 成功")
    print("关键处理结果: vector、text、metadata 均已保存并从同一个 Point 回读")
    print("最终输出: 单个 Chunk 已写入并可按 ID 查询")


# 增加于阶段 7.3：提供单个 Chunk 写入 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
