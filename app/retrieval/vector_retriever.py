"""阶段 8.1–8.2：基于 Qdrant 的向量检索和标准结果对象。"""

from collections.abc import Mapping
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.embedding.embeddings import embed_query
from app.security import User
from app.security.permissions import build_permission_filter
from app.vectorstore.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    get_qdrant_client,
)


# 增加于阶段 8.2：定义统一的检索结果对象。
# 修改于阶段 10.1：显式保留 Citation 所需的 Chunk ID。
@dataclass(frozen=True)
class RetrievedChunk:
    """表示一个带正文、相似度和完整来源信息的检索 Chunk。"""

    content: str
    score: float
    source: str | None
    page: int | str | None
    metadata: dict[str, object]
    chunk_id: str | None = None


# 增加于阶段 8.2：把 Qdrant 返回的 ScoredPoint 转换为标准对象。
def _to_retrieved_chunk(point: object) -> RetrievedChunk:
    """将 Qdrant 的单条 ScoredPoint 转换为 RetrievedChunk。

    实现方式：读取 Point 的 payload，校验正文、Metadata 和相似度分数，再从
    Metadata 中提取 source_file/source 和 page_label/page；同时保留完整 Metadata
    副本，确保后续 RAG 可以继续使用业务字段和 Chunk ID。

    参数：
        point: Qdrant query_points() 返回的单条 ScoredPoint 对象。

    返回：
        RetrievedChunk：包含正文、相似度分数、来源、页码和完整 Metadata 的标准对象。

    异常：
        RuntimeError: Point 缺少合法 payload、正文、source、page、chunk_id 或数值
            相似度分数时抛出。
    """
    payload = getattr(point, "payload", None)
    if not isinstance(payload, Mapping):
        raise RuntimeError("检索 Point 缺少合法 payload")

    content = payload.get("text")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("检索 Point 缺少非空正文")

    metadata_value = payload.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise RuntimeError("检索 Point 的 metadata 不是对象")
    metadata = dict(metadata_value)
    chunk_id_value = payload.get("chunk_id", metadata.get("chunk_id"))
    chunk_id = (
        chunk_id_value.strip()
        if isinstance(chunk_id_value, str) and chunk_id_value.strip()
        else None
    )
    if chunk_id is None:
        raise RuntimeError("检索 Point 缺少非空 chunk_id")
    metadata["chunk_id"] = chunk_id

    score = getattr(point, "score", None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("检索 Point 缺少数值相似度分数")

    source_value = metadata.get("source_file") or metadata.get("source")
    source = source_value.strip() if isinstance(source_value, str) else None
    if not source:
        raise RuntimeError("检索 Point 缺少非空 source")
    page_value = metadata.get("page_label", metadata.get("page"))
    page = (
        page_value.strip()
        if isinstance(page_value, str)
        else page_value
        if isinstance(page_value, int) and not isinstance(page_value, bool)
        else None
    )
    if page is None or (isinstance(page, str) and not page):
        raise RuntimeError("检索 Point 缺少合法 page")

    return RetrievedChunk(
        content=content,
        score=float(score),
        source=source,
        page=page,
        metadata=metadata,
        chunk_id=chunk_id,
    )


# 增加于阶段 8.1：实现从 Query 到 Top K Chunk 的向量检索。
# 修改于阶段 8.2：将 Qdrant 结果统一转换为 RetrievedChunk 对象。
# 修改于阶段 12.3：检索前必须应用基于 User.role 的权限过滤器。
def search(
    query: str,
    top_k: int = 5,
    client: QdrantClient | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    user: User | None = None,
) -> list[RetrievedChunk]:
    """使用 Query 向量从 Qdrant 召回 Top K 个相关 Chunk。

    实现方式：校验 Query、top_k、Collection 名称和 User 后，使用与文档相同的
    Embedding 封装生成 Query 向量，先根据 User.role 构造 Qdrant Metadata Filter，
    再调用 query_points() 进行带权限条件的 Cosine 相似度检索；返回的每条
    ScoredPoint 由本模块转换为统一 RetrievedChunk，保留正文、分数、来源、页码和
    业务 Metadata，不把权限控制交给 Prompt。

    参数：
        query: 用户自然语言问题，必须是非空字符串。
        top_k: 最多召回的 Chunk 数量，必须是正整数，默认 5。
        client: 可选的 QdrantClient；未提供时连接默认本地 Qdrant 服务。
        collection_name: 检索目标 Collection 名称，默认 enterprise_knowledge。
        user: 当前请求用户，必须是阶段 12.1 的 User 对象；缺少用户时拒绝无过滤检索。

    返回：
        list[RetrievedChunk]：按 Qdrant 相似度从高到低返回的检索结果；没有匹配
            Point 时返回空列表，结果数量不会超过 top_k。

    异常：
        TypeError: query 不是字符串或 top_k 不是整数时抛出。
        ValueError: query 为空、top_k 不为正数、Collection 名称为空或 user 缺失时抛出。
        TypeError: user 不是 User 对象时由权限过滤器抛出。
        RuntimeError: Qdrant Point payload 不完整时抛出。
        Exception: Embedding、Qdrant 服务不可达或检索请求失败时透传底层异常。
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string")
    if not collection_name.strip():
        raise ValueError("collection_name must not be empty")
    if user is None:
        raise ValueError("user is required for permission-filtered retrieval")
    permission_filter = build_permission_filter(user)

    query_vector = embed_query(query)
    if not query_vector:
        raise RuntimeError("Query Embedding 为空，无法执行检索")

    qdrant_client = client or get_qdrant_client()
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=permission_filter,
        with_payload=True,
        with_vectors=False,
    )
    return [_to_retrieved_chunk(point) for point in response.points]
