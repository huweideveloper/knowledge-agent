"""阶段 7.3–7.6：Chunk 写入和文档版本更新的封装。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Record

from app.embedding.embeddings import embed_documents
from app.ingestion.cleaner import clean_text
from app.ingestion.loader import load_directory, load_pdf
from app.ingestion.splitter import split_documents
from app.vectorstore.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    delete_points_by_document_id,
    ensure_collection,
    retrieve_point,
    to_qdrant_point_id,
    upsert_point,
)


# 增加于阶段 7.3：将单个 Chunk 的正文、向量和 Metadata 组装为一个 Point。
# 修改于阶段 7.4：将业务 Chunk ID 映射为稳定的 Qdrant UUID，并保留原始 ID。
def write_chunk(
    client: QdrantClient,
    chunk_id: str,
    text: str,
    vector: Sequence[float],
    metadata: Mapping[str, object],
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """将一个 Chunk 写入指定 Qdrant Collection。

    实现方式：校验 Chunk ID 和正文后，把正文放入 payload 的 text 字段，把业务
    Metadata 放入 payload 的 metadata 字段，再委托 vectorstore 层将向量和 payload
    作为一个 Point upsert。该函数只处理单个 Chunk，不执行批量遍历或检索排序。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        chunk_id: Chunk 对应的稳定 Point ID，不能为空。
        text: Chunk 正文，不能为空字符串。
        vector: 该 Chunk 的 Embedding 向量，必须是非空数值序列。
        metadata: Chunk 的业务 Metadata 映射，可以为空但必须是 Mapping。
        collection_name: 目标 Collection 名称，默认为 enterprise_knowledge。

    返回：
        无返回值；Qdrant upsert 成功表示该 Chunk 已写入。

    异常：
        TypeError: Chunk、向量、Metadata 或 Collection 参数类型不符合要求时抛出。
        ValueError: Chunk ID、正文或 Collection 名称为空时抛出。
        Exception: Qdrant 服务不可达、Collection 不存在或写入失败时透传底层异常。
    """
    if not isinstance(chunk_id, str):
        raise TypeError("chunk_id must be a string")
    if not chunk_id.strip():
        raise ValueError("chunk_id must not be empty")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    qdrant_point_id = to_qdrant_point_id(chunk_id)
    upsert_point(
        client=client,
        collection_name=collection_name,
        point_id=qdrant_point_id,
        vector=vector,
        payload={
            "chunk_id": chunk_id,
            "text": text,
            "metadata": dict(metadata),
        },
    )


# 增加于阶段 7.3：按 Chunk ID 查询单个 Chunk 的完整 Qdrant Record。
# 修改于阶段 7.4：查询时使用与写入一致的稳定 UUID 映射。
def read_chunk(
    client: QdrantClient,
    chunk_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Record:
    """按 Chunk ID 查询保存的正文、Metadata 和向量。

    实现方式：委托 vectorstore 层调用 Qdrant retrieve API，并将单个 Record 原样
    返回给调用方；查询要求同时带回 payload 和 vector，以便验收写入链路的三个
    数据部分。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        chunk_id: 要查询的 Chunk/Point 字符串 ID，不能为空。
        collection_name: Chunk 所在的 Collection 名称，默认为 enterprise_knowledge。

    返回：
        Record：包含 id、payload 和 vector 的 Qdrant 记录。

    异常：
        TypeError: Chunk ID 或 Collection 名称类型不符合要求时抛出。
        ValueError: Chunk ID 或 Collection 名称为空时抛出。
        LookupError: Qdrant 中不存在指定 Chunk ID 时抛出。
        Exception: Qdrant 服务不可达或查询失败时透传底层异常。
    """
    return retrieve_point(
        client=client,
        collection_name=collection_name,
        point_id=to_qdrant_point_id(chunk_id),
    )


# 增加于阶段 7.4：记录批量入库各处理阶段的数量结果。
@dataclass(frozen=True)
class IngestionResult:
    """描述一次批量入库的输入规模和写入结果。"""

    target_pdf_count: int
    documents_loaded: int
    chunks_created: int
    vectors_created: int
    points_written: int


# 增加于阶段 7.6：集中校验单条业务 Metadata JSON 记录。
# 修改于阶段 12.2：要求文档声明非空的 allowed_roles 权限列表。
def _validate_metadata_record(
    record: object,
    metadata_path: Path,
) -> dict[str, object]:
    """校验单条 Metadata 记录并返回其字典内容。

    实现方式：要求 JSON 顶层是对象，并检查 document_id、title 都是非空字符串，
    同时要求 allowed_roles 是至少包含一个非空字符串的列表；该校验同时服务于
    目录批量入库和单文档版本更新，避免缺少权限的 Metadata 在删除旧版本后才被发现。

    参数：
        record: 从 JSON 解码得到的任意 Python 对象。
        metadata_path: 当前记录对应的 JSON 路径，用于生成可定位的错误信息。

    返回：
        dict[str, object]：通过校验的业务 Metadata 字典，保留其他可选字段。

    异常：
        ValueError: record 不是对象，必填字段不是非空字符串，或 allowed_roles 不是
            非空字符串列表时抛出。
    """
    if not isinstance(record, dict):
        raise ValueError(f"Metadata must be an object: {metadata_path}")
    for field_name in ("document_id", "title"):
        field_value = record.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                f"Metadata requires non-empty string {field_name}: {metadata_path}"
            )
    allowed_roles = record.get("allowed_roles")
    if not isinstance(allowed_roles, list) or not allowed_roles or any(
        not isinstance(role, str) or not role.strip() for role in allowed_roles
    ):
        raise ValueError(
            f"Metadata requires non-empty string list allowed_roles: {metadata_path}"
        )
    return record


# 增加于阶段 7.4：读取并校验目录中的业务 Metadata JSON。
# 修改于阶段 7.6：复用单条 Metadata 校验逻辑。
# 修改于阶段 12.2：加载时一并校验文档 allowed_roles 权限信息。
def _load_metadata_records(metadata_directory: Path) -> list[dict[str, object]]:
    """读取 Metadata 目录中的全部 JSON 记录。

    实现方式：按文件名排序读取 JSON，调用单条 Metadata 校验逻辑确认每条记录是
    对象且包含 document_id、title 和 allowed_roles，返回原始字典列表供后续按 PDF
    文件名或标题匹配。排序保证同一输入目录下匹配顺序稳定。

    参数：
        metadata_directory: Metadata JSON 所在目录，必须是目录。

    返回：
        list[dict[str, object]]：按 JSON 文件名排序的业务 Metadata 记录列表。

    异常：
        NotADirectoryError: metadata_directory 不存在或不是目录时抛出。
        ValueError: JSON 不是对象，或缺少/错误的 document_id、title、allowed_roles
            时抛出。
        json.JSONDecodeError: JSON 文件格式错误时抛出。
    """
    if not metadata_directory.is_dir():
        raise NotADirectoryError(f"Metadata directory not found: {metadata_directory}")

    records: list[dict[str, object]] = []
    for metadata_path in sorted(metadata_directory.glob("*.json")):
        records.append(
            _validate_metadata_record(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                metadata_path,
            )
        )
    return records


# 增加于阶段 7.4：按 PDF 来源或标题匹配业务 Metadata。
def _find_metadata_record(
    document: Document,
    metadata_records: list[dict[str, object]],
) -> dict[str, object]:
    """为一个加载后的 PDF 页面找到对应的业务 Metadata。

    实现方式：优先使用 Metadata 中显式的 source_file 与 PDF 文件名匹配；没有
    source_file 时，将 PDF 的 title 与 Metadata title 去除空白后比较，兼容 PDF
    元数据和 JSON 标题之间的排版空格差异。

    参数：
        document: Loader 产生的单页 Document，必须包含 source 和可匹配的 title。
        metadata_records: 已读取并校验的业务 Metadata 记录列表。

    返回：
        dict[str, object]：与该 PDF 对应的业务 Metadata 字典。

    异常：
        ValueError: Document 缺少 source/title，或没有唯一匹配的 Metadata 时抛出。
    """
    source_value = document.metadata.get("source")
    title_value = document.metadata.get("title")
    if not source_value or not title_value:
        raise ValueError("Document metadata requires source and title")

    source_name = Path(str(source_value)).name
    source_matches = [
        record
        for record in metadata_records
        if Path(str(record.get("source_file", ""))).name == source_name
    ]
    if len(source_matches) == 1:
        return source_matches[0]

    normalized_title = "".join(str(title_value).split()).casefold()
    title_matches = [
        record
        for record in metadata_records
        if "".join(str(record.get("title", "")).split()).casefold()
        == normalized_title
    ]
    if len(title_matches) != 1:
        raise ValueError(f"No unique Metadata match for PDF: {source_name}")
    return title_matches[0]


# 增加于阶段 7.6：准备待写入的清洗后 Document，供批量和版本更新共用。
def _prepare_documents(
    documents: list[Document],
    metadata_records: list[dict[str, object]],
) -> list[Document]:
    """合并业务 Metadata 并清洗已加载的 Document 正文。

    实现方式：为每页 Document 匹配 source_file 或标题对应的业务 Metadata，合并
    两侧元数据并补充 source_file，再调用统一清洗流程生成新的 Document 列表；不
    修改 Loader 返回的原始对象。

    参数：
        documents: Loader 产生的 PDF 页面 Document 列表。
        metadata_records: 已通过校验的业务 Metadata 记录列表。

    返回：
        list[Document]：完成 Metadata 合并和正文清洗的 Document 列表。

    异常：
        ValueError: 某个 Document 无法唯一匹配业务 Metadata 时抛出。
    """
    cleaned_documents: list[Document] = []
    for document in documents:
        business_metadata = _find_metadata_record(document, metadata_records)
        merged_metadata = dict(document.metadata)
        merged_metadata.update(business_metadata)
        merged_metadata.setdefault("source_file", Path(str(document.metadata["source"])).name)
        cleaned_documents.append(
            Document(
                page_content=clean_text(document.page_content),
                metadata=merged_metadata,
            )
        )
    return cleaned_documents


# 增加于阶段 7.6：复用完整的切块、向量化和 Point 写入流程。
def _ingest_loaded_documents(
    client: QdrantClient,
    documents: list[Document],
    metadata_records: list[dict[str, object]],
    target_pdf_count: int,
    collection_name: str,
) -> IngestionResult:
    """把已加载的文档执行清洗、切块、向量化和 Qdrant 写入。

    实现方式：先合并业务 Metadata 并清洗正文，再按 V1 切块规则生成稳定 chunk_id，
    使用同一 Embedding 模型批量生成向量，校验维度后创建或校验 Collection，最后
    逐个 upsert Point。该流程同时支持全量入库和单个新版本写入。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        documents: 已由 Loader 加载的 PDF 页面 Document 列表。
        metadata_records: 与 documents 对应且已通过校验的业务 Metadata 列表。
        target_pdf_count: 本次处理的 PDF 文件数量，必须为正整数。
        collection_name: 写入的 Qdrant Collection 名称，不能为空。

    返回：
        IngestionResult：包含 PDF、Document、Chunk、向量和写入 Point 数量。

    异常：
        TypeError: target_pdf_count 不是整数时抛出。
        ValueError: target_pdf_count 不为正数、无 Chunk、Embedding 为空、Metadata
            无法匹配或 Collection 配置不匹配时抛出。
        RuntimeError: 向量数量或处理结果与 Chunk 不匹配时抛出。
        Exception: 向量化、Qdrant 服务或写入请求失败时透传底层异常。
    """
    if isinstance(target_pdf_count, bool) or not isinstance(target_pdf_count, int):
        raise TypeError("target_pdf_count must be an integer")
    if target_pdf_count <= 0:
        raise ValueError("target_pdf_count must be greater than zero")

    cleaned_documents = _prepare_documents(documents, metadata_records)
    chunks = split_documents(cleaned_documents)
    if not chunks:
        raise ValueError("No chunks were created from target PDFs")

    vectors = embed_documents([chunk.page_content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding vector count does not match chunk count")
    vector_dimensions = {len(vector) for vector in vectors}
    if len(vector_dimensions) != 1:
        raise ValueError("Embedding vectors must have the same dimension")
    vector_size = vector_dimensions.pop()
    if vector_size <= 0:
        raise ValueError("Embedding vector dimension must be greater than zero")

    ensure_collection(
        client,
        vector_size=vector_size,
        collection_name=collection_name,
    )
    for chunk, vector in zip(chunks, vectors):
        chunk_id = chunk.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Each chunk must have a non-empty string chunk_id")
        write_chunk(
            client=client,
            chunk_id=chunk_id,
            text=chunk.page_content,
            vector=vector,
            metadata=chunk.metadata,
            collection_name=collection_name,
        )

    return IngestionResult(
        target_pdf_count=target_pdf_count,
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        vectors_created=len(vectors),
        points_written=len(chunks),
    )


# 增加于阶段 7.4：执行 Loader 到 Qdrant 的完整批量入库流程。
# 修改于阶段 7.6：委托共用的文档处理和写入流程。
def ingest_documents(
    client: QdrantClient,
    source_directory: str | Path = Path("data/raw"),
    metadata_directory: str | Path = Path("data/metadata"),
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> IngestionResult:
    """批量执行 Loader、Cleaner、Splitter、Embedding 和 Qdrant 入库流程。

    实现方式：先加载 source_directory 中的全部 PDF 并确认没有目标文件被跳过，
    再读取并匹配业务 Metadata，最后委托共用流程清洗、切块、向量化并逐个 upsert。
    每个业务 chunk_id 通过稳定 UUID 映射为 Qdrant Point ID，重复执行只更新原 Point。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        source_directory: 目标 PDF 目录，默认 data/raw。
        metadata_directory: 业务 Metadata JSON 目录，默认 data/metadata。
        collection_name: 写入的 Qdrant Collection 名称，默认 enterprise_knowledge。

    返回：
        IngestionResult：包含目标 PDF、加载 Document、Chunk、向量和写入 Point 数量。

    异常：
        NotADirectoryError: PDF 或 Metadata 目录不存在或不是目录时抛出。
        ValueError: Metadata 不完整、无法匹配、Embedding 为空或 Collection 配置
            不匹配时抛出。
        RuntimeError: 目标 PDF 被跳过、向量数量不匹配或服务处理结果异常时抛出。
        Exception: Qdrant 服务不可达、向量化或写入失败时透传底层异常。
    """
    source_path = Path(source_directory)
    metadata_path = Path(metadata_directory)
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source directory not found: {source_path}")

    pdf_paths = sorted(source_path.glob("*.pdf"))
    documents = load_directory(source_path)
    loaded_sources = {
        Path(str(document.metadata.get("source"))).name
        for document in documents
    }
    missing_sources = {path.name for path in pdf_paths} - loaded_sources
    if missing_sources:
        missing_text = ", ".join(sorted(missing_sources))
        raise RuntimeError(f"目标 PDF 未能加载：{missing_text}")

    return _ingest_loaded_documents(
        client=client,
        documents=documents,
        metadata_records=_load_metadata_records(metadata_path),
        target_pdf_count=len(pdf_paths),
        collection_name=collection_name,
    )


# 增加于阶段 7.6：删除旧版本后重新生成并写入指定新版本文档。
def replace_document_version(
    client: QdrantClient,
    old_document_id: str,
    new_source_file: str | Path,
    new_metadata_file: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> IngestionResult:
    """用新版本文档替换旧版本文档在 Qdrant 中的全部 chunks。

    实现方式：先校验新 PDF 和 Metadata，并加载原始页面；随后按旧 document_id
    删除全部旧 Point，删除确认完成后再执行清洗、切块、Embedding 和新 Point 写入。
    新版本使用自己的 document_id 生成稳定 chunk_id，因此后续重复更新仍保持幂等。

    参数：
        client: 已创建并可访问 Qdrant 服务的 QdrantClient 实例。
        old_document_id: 要废弃的旧文档版本 ID，必须是非空字符串。
        new_source_file: 新版本 PDF 文件路径，必须是存在的普通文件。
        new_metadata_file: 新版本 Metadata JSON 路径，必须是存在的普通文件。
        collection_name: 目标 Qdrant Collection 名称，默认为 enterprise_knowledge。

    返回：
        IngestionResult：新版本实际加载、切分、向量化和写入的数量结果。

    异常：
        TypeError: ID 或 Collection 名称类型不符合要求时抛出。
        ValueError: ID 为空、新 Metadata 仍使用旧 document_id、Metadata 无效或
            新文档没有可写入 Chunk 时抛出。
        FileNotFoundError: 新 PDF 或 Metadata 文件不存在时抛出。
        RuntimeError: 旧 Point 删除未完成、Embedding 数量不匹配或处理结果异常时抛出。
        Exception: PDF 解析、向量化、Qdrant 服务或写入请求失败时透传底层异常。
    """
    if not isinstance(old_document_id, str):
        raise TypeError("old_document_id must be a string")
    if not old_document_id.strip():
        raise ValueError("old_document_id must not be empty")

    source_path = Path(new_source_file)
    metadata_path = Path(new_metadata_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"New PDF not found: {source_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"New Metadata not found: {metadata_path}")

    metadata_record = _validate_metadata_record(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        metadata_path,
    )
    new_document_id = metadata_record["document_id"]
    if new_document_id == old_document_id:
        raise ValueError("New document_id must differ from old_document_id")

    documents = load_pdf(source_path)
    delete_points_by_document_id(
        client=client,
        document_id=old_document_id,
        collection_name=collection_name,
    )
    return _ingest_loaded_documents(
        client=client,
        documents=documents,
        metadata_records=[metadata_record],
        target_pdf_count=1,
        collection_name=collection_name,
    )
