"""阶段 7.2：演示创建与 Embedding 维度一致的 Qdrant Collection。"""

import sys
from pathlib import Path


# 增加于阶段 7.2：支持从项目根目录直接运行 Collection 创建 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import (
    embed_documents,
    embed_query,
    validate_embedding_dimensions,
)
from app.vectorstore.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    get_qdrant_client,
    ensure_collection,
)


# 增加于阶段 7.2：定义用于确定 Embedding 维度的中文示例输入。
SAMPLE_TEXT = "上海出差住宿上限为普通员工 600 元/晚。"
SAMPLE_QUERY = "上海出差住宿标准是多少？"


# 增加于阶段 7.2：运行 Collection 创建和维度一致性验收 Demo。
def run_demo() -> None:
    """生成 Embedding 维度并创建或校验 enterprise_knowledge Collection。

    实现方式：调用正式文档和 Query Embedding 接口，使用阶段 6.3 的维度校验
    得到当前向量长度，再连接阶段 7.1 的本地 Qdrant 服务并调用正式 Collection
    封装。最后读取 Collection 配置，比较向量维度和 Cosine 距离并打印结果。
    本 Demo 不写入 Point，写入 Point 属于阶段 7.3。

    参数：
        无入参；Embedding 维度来自内置中文示例文本和 Query。

    返回：
        无返回值；Collection 名称、维度、距离和验收结果通过标准输出打印。

    异常：
        RuntimeError: Qdrant 服务不可访问，或最终配置验收失败时抛出。
        ValueError: Embedding 维度为空/不一致，或已有 Collection 配置不匹配时抛出。
        TypeError: 正式 Embedding 或 Collection 接口收到错误类型时抛出。
    """
    document_vectors = embed_documents([SAMPLE_TEXT])
    query_vector = embed_query(SAMPLE_QUERY)
    embedding_dimension = validate_embedding_dimensions(
        document_vectors,
        query_vector,
    )

    client = get_qdrant_client()
    was_created = ensure_collection(
        client,
        vector_size=embedding_dimension,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    collection_info = client.get_collection(collection_name=DEFAULT_COLLECTION_NAME)
    vector_config = collection_info.config.params.vectors
    collection_dimension = vector_config.size
    distance = str(vector_config.distance)

    if collection_dimension != embedding_dimension or distance != "Cosine":
        raise RuntimeError("Collection 配置与 Embedding 不一致")

    print("=== 阶段 7.2 Collection 创建 Demo ===")
    print(f"Collection 名称: {DEFAULT_COLLECTION_NAME}")
    print(f"Embedding 向量维度: {embedding_dimension}")
    print(f"Collection 向量维度: {collection_dimension}")
    print(f"距离度量: {distance}")
    print(f"创建状态: {'已创建' if was_created else '已存在且配置一致'}")
    print("维度一致性: 通过")
    print("关键处理结果: Collection 已准备好接收后续阶段的 Point")
    print("最终输出: Collection 创建并配置完成")


# 增加于阶段 7.2：提供 Collection 创建 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
