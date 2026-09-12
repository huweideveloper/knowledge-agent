"""阶段 7.1–7.6：Qdrant 服务、Collection 和 Point 操作封装。"""

from collections.abc import Mapping, Sequence
import time
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models


# 增加于阶段 7.1：定义本地 Qdrant REST 服务地址。
DEFAULT_QDRANT_URL = "http://localhost:6333"

# 增加于阶段 7.2：定义 V1 使用的 Collection 名称。
DEFAULT_COLLECTION_NAME = "enterprise_knowledge"


# 增加于阶段 7.1：创建连接本地 Qdrant 服务的 Python 客户端。
# 修改于阶段 7.1：优先使用 Compose 暴露的 gRPC 端口，规避当前环境 REST 客户端的 502。
# 修改于阶段 7.4：关闭本地服务版本探测，避免客户端与服务小版本差异产生误导性警告。
def get_qdrant_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    """创建一个指向指定 Qdrant 服务地址的 Python 客户端。

    实现方式：校验 URL 为非空字符串后，使用 qdrant-client 的 QdrantClient
    创建客户端，并优先通过同一 Qdrant 服务的 gRPC 端口访问，同时关闭本地版本
    兼容性探测；本函数只负责建立客户端对象，不在这里创建 Collection 或写入任何
    向量。

    参数：
        url: Qdrant REST 服务地址，默认为本地 Docker 暴露的
            http://localhost:6333。

    返回：
        QdrantClient：可调用 Qdrant 服务 API 的同步 Python 客户端。

    异常：
        TypeError: url 不是字符串时抛出。
        ValueError: url 为空字符串或只包含空白字符时抛出。
    """
    if not isinstance(url, str):
        raise TypeError("Qdrant URL must be a string")
    if not url.strip():
        raise ValueError("Qdrant URL must not be empty")

    return QdrantClient(
        url=url,
        prefer_grpc=True,
        check_compatibility=False,
    )


# 增加于阶段 7.1：通过集合列表请求验证 Qdrant 服务连接。
def list_collection_names(client: QdrantClient) -> list[str]:
    """请求 Qdrant 集合列表并返回集合名称。

    实现方式：调用客户端的 get_collections() API，将 Qdrant 返回的集合对象转换
    为简单字符串列表；该请求既验证服务可达，也不会改变服务数据。

    参数：
        client: 已创建的 QdrantClient 实例，必须能够调用 get_collections()。

    返回：
        list[str]：当前 Qdrant 服务中的 Collection 名称列表；没有 Collection
            时返回空列表。

    异常：
        AttributeError: client 不具备 get_collections() 方法时抛出。
        Exception: Qdrant 服务不可达或 API 请求失败时透传客户端异常。
    """
    response = client.get_collections()
    return [collection.name for collection in response.collections]


# 增加于阶段 7.2：创建或校验与 Embedding 维度匹配的 Qdrant Collection。
def ensure_collection(
    client: QdrantClient,
    vector_size: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> bool:
    """创建指定名称的 Cosine Collection，或校验已有 Collection 配置。

    实现方式：先校验 Collection 名称和向量维度；如果 Collection 不存在，则
    使用 qdrant-client 创建单向量配置，向量距离固定为 Cosine。如果 Collection
    已存在，则读取其配置并要求向量维度和距离都一致，避免使用破坏性的重建操作。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        vector_size: 当前 Embedding 输出的向量维度，必须为正整数。
        collection_name: 要创建或校验的 Collection 名称，默认为
            enterprise_knowledge，不能为空。

    返回：
        bool：本次调用新创建 Collection 时返回 True；Collection 已存在且配置
            校验通过时返回 False。

    异常：
        TypeError: vector_size 不是整数，或 collection_name 不是字符串时抛出。
        ValueError: vector_size 不为正数、名称为空，或已有 Collection 的向量
            维度/距离与当前 V1 配置不一致时抛出。
        Exception: Qdrant 服务不可达或创建、读取 Collection 失败时透传客户端异常。
    """
    if isinstance(vector_size, bool) or not isinstance(vector_size, int):
        raise TypeError("vector_size must be an integer")
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than zero")
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return True

    collection_info = client.get_collection(collection_name=collection_name)
    vector_config = collection_info.config.params.vectors
    if vector_config.size != vector_size:
        raise ValueError(
            f"Collection 向量维度不匹配：已有 {vector_config.size}，需要 {vector_size}"
        )
    if vector_config.distance != models.Distance.COSINE:
        raise ValueError(
            f"Collection 距离度量不匹配：已有 {vector_config.distance}，需要 Cosine"
        )

    return False


# 增加于阶段 7.4：将业务 Chunk ID 稳定转换为 Qdrant 支持的 UUID 字符串。
def to_qdrant_point_id(point_id: str) -> str:
    """把业务层字符串 ID 转换为可用于 Qdrant 的稳定 Point ID。

    实现方式：合法 UUID 字符串直接规范化后返回；其他业务字符串使用标准库
    uuid5 和固定命名空间生成确定性 UUID。这样阶段 5 生成的可读 chunk_id 可以
    保留在业务 payload 中，同时满足 Qdrant gRPC 对字符串 Point ID 的格式要求。

    参数：
        point_id: 业务层 Point 或 Chunk 字符串 ID，不能为空。

    返回：
        str：可用于 Qdrant PointStruct 的规范化 UUID 字符串；同一输入始终返回
            同一结果。

    异常：
        TypeError: point_id 不是字符串时抛出。
        ValueError: point_id 为空字符串时抛出。
    """
    if not isinstance(point_id, str):
        raise TypeError("point_id must be a string")
    if not point_id.strip():
        raise ValueError("point_id must not be empty")

    try:
        return str(UUID(point_id))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, point_id))


# 增加于阶段 7.5：读取 Collection 的精确 Point 数量用于幂等验收。
def count_points(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """返回指定 Collection 中的精确 Point 数量。

    实现方式：调用 qdrant-client 的 count API 并启用 exact=True，读取服务端实际
    点数；该请求只读，不创建、更新或删除任何 Point，供重复入库前后进行数量比较。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        collection_name: 要统计的 Collection 名称，默认为 enterprise_knowledge。

    返回：
        int：Collection 当前保存的精确 Point 数量，空 Collection 返回 0。

    异常：
        TypeError: collection_name 不是字符串时抛出。
        ValueError: collection_name 为空时抛出。
        Exception: Qdrant 服务不可达或统计请求失败时透传客户端异常。
    """
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    return client.count(collection_name=collection_name, exact=True).count


# 增加于阶段 7.6：按业务 document_id 统计某个文档版本的 Point 数量。
def count_points_by_document_id(
    client: QdrantClient,
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """返回指定文档版本在 Collection 中保存的精确 Point 数量。

    实现方式：使用 Qdrant payload 的嵌套字段 metadata.document_id 建立过滤器，
    再调用 exact=True 的 count API；该函数只读，用于文档更新前后确认旧版本和新
    版本的实际存储状态。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        document_id: 业务文档版本 ID，必须是非空字符串。
        collection_name: 要统计的 Collection 名称，默认为 enterprise_knowledge。

    返回：
        int：匹配 document_id 的精确 Point 数量，没有匹配时返回 0。

    异常：
        TypeError: document_id 或 collection_name 不是字符串时抛出。
        ValueError: document_id 或 collection_name 为空时抛出。
        Exception: Qdrant 服务不可达或统计请求失败时透传客户端异常。
    """
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a string")
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )
    return client.count(
        collection_name=collection_name,
        count_filter=document_filter,
        exact=True,
    ).count


# 增加于阶段 7.6：删除指定业务文档版本的全部 Point，并确认删除结果。
def delete_points_by_document_id(
    client: QdrantClient,
    document_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> int:
    """删除指定文档版本的全部 Point，并返回实际删除数量。

    实现方式：先按 payload 中的 metadata.document_id 精确统计旧 Point，调用
    Qdrant 的 FilterSelector 删除匹配记录并等待服务确认，最后再次统计并要求
    数量归零。删除范围只包含指定 document_id，不影响其他文档版本或业务数据。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        document_id: 要废弃的业务文档版本 ID，必须是非空字符串。
        collection_name: 要删除 Point 的 Collection 名称，默认为
            enterprise_knowledge。

    返回：
        int：删除前匹配到的 Point 数量；没有旧版本时返回 0。

    异常：
        TypeError: document_id 或 collection_name 不是字符串时抛出。
        ValueError: document_id 或 collection_name 为空时抛出。
        RuntimeError: Qdrant 确认删除后仍有匹配 Point 时抛出。
        Exception: Qdrant 服务不可达或删除请求失败时透传客户端异常。
    """
    before_count = count_points_by_document_id(
        client=client,
        document_id=document_id,
        collection_name=collection_name,
    )
    if before_count == 0:
        return 0

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=document_filter),
        wait=True,
    )
    remaining_count = count_points_by_document_id(
        client=client,
        document_id=document_id,
        collection_name=collection_name,
    )
    if remaining_count != 0:
        raise RuntimeError(
            f"文档版本删除未完成：{document_id} 仍有 {remaining_count} 个 Point"
        )
    return before_count


# 增加于阶段 7.3：将一个带向量和 payload 的 Point 写入 Qdrant。
def upsert_point(
    client: QdrantClient,
    collection_name: str,
    point_id: str,
    vector: Sequence[float],
    payload: Mapping[str, object],
) -> None:
    """向指定 Collection 写入一个包含 vector 和 payload 的 Point。

    实现方式：校验 Collection、Point ID、向量和 payload 后构造 qdrant-client 的
    PointStruct，调用 upsert 并等待服务确认。payload 保存正文和业务 Metadata，
    向量保留为 Point 的独立 vector 字段；相同 ID 重复写入时按 Qdrant upsert
    语义更新该 Point，不创建额外记录。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        collection_name: 目标 Collection 名称，不能为空。
        point_id: Point 的稳定字符串 ID，不能为空。
        vector: 要保存的 Embedding 向量，必须是非空数值序列。
        payload: 要保存的正文和 Metadata 映射，必须是 Mapping；函数会复制一份
            映射，避免调用方后续修改影响本次请求。

    返回：
        无返回值；Qdrant 返回成功表示 Point 已写入或更新。

    异常：
        TypeError: ID、名称、向量元素或 payload 类型不符合要求时抛出。
        ValueError: Collection/ID 为空或向量为空时抛出。
        Exception: Qdrant 服务不可达、Collection 不存在或 upsert 失败时透传客户端异常。
    """
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")
    if not isinstance(point_id, str):
        raise TypeError("point_id must be a string")
    if not point_id.strip():
        raise ValueError("point_id must not be empty")
    if isinstance(vector, str):
        raise TypeError("vector must be a numeric sequence")
    vector_values = list(vector)
    if not vector_values:
        raise ValueError("vector must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in vector_values
    ):
        raise TypeError("vector must contain numeric values")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector_values,
                payload=dict(payload),
            )
        ],
        wait=True,
    )


# 增加于阶段 7.3：按 Point ID 查询刚写入的完整记录。
def retrieve_point(
    client: QdrantClient,
    collection_name: str,
    point_id: str,
) -> object:
    """按 ID 查询一个 Point，并要求返回其 vector 和 payload。

    实现方式：调用 Qdrant retrieve API，开启 with_payload 和 with_vectors，
    从返回列表中取出唯一记录；没有匹配记录时抛出 LookupError，避免调用方把
    空查询误认为写入成功。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        collection_name: 查询所在的 Collection 名称，不能为空。
        point_id: 要查询的 Point 字符串 ID，不能为空。

    返回：
        object：qdrant-client 返回的单个 Record，包含 id、vector 和 payload 字段。

    异常：
        TypeError: Collection 名称或 Point ID 不是字符串时抛出。
        ValueError: Collection 名称或 Point ID 为空时抛出。
        LookupError: Qdrant 中不存在指定 ID 时抛出。
        Exception: Qdrant 服务不可达或查询失败时透传客户端异常。
    """
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")
    if not isinstance(point_id, str):
        raise TypeError("point_id must be a string")
    if not point_id.strip():
        raise ValueError("point_id must not be empty")

    records = client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    if not records:
        raise LookupError(f"Point 不存在：{point_id}")
    return records[0]


# 增加于阶段 7.1：等待 Docker 中的 Qdrant 服务完成启动并接受请求。
def wait_for_qdrant(
    client: QdrantClient,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.5,
) -> list[str]:
    """轮询 Qdrant 服务直到可访问或超过等待时间。

    实现方式：在限定时间内重复调用集合列表接口；服务尚未启动时短暂等待后
    重试，服务可访问时立即返回集合名称，超时则将最后一次客户端异常包装为
    RuntimeError。轮询只执行只读请求，不创建 Collection 或修改服务数据。

    参数：
        client: 已创建的 QdrantClient 实例。
        timeout_seconds: 最长等待秒数，必须大于 0。
        poll_interval_seconds: 两次请求之间的等待秒数，必须大于 0。

    返回：
        list[str]：Qdrant 服务可访问时的 Collection 名称列表。

    异常：
        ValueError: timeout_seconds 或 poll_interval_seconds 不大于 0 时抛出。
        RuntimeError: 在超时时间内 Qdrant 始终无法访问时抛出。
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return list_collection_names(client)
        except Exception as error:
            last_error = error
            time.sleep(poll_interval_seconds)

    try:
        return list_collection_names(client)
    except Exception as error:
        last_error = error

    raise RuntimeError("Qdrant 服务在限定时间内未能访问") from last_error
