from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 增加于阶段 5.5：集中定义 V1 Chunking 的默认基线参数。
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


# 增加于阶段 5.1：使用固定长度策略切分 Document。
# 修改于阶段 5.2：为每个 Chunk 增加可追溯的唯一 ID。
# 修改于阶段 5.3：明确保留原始 Document 的业务和权限 Metadata。
# 修改于阶段 5.5：允许调用方配置 V1 基线的切块大小和重叠长度。
def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """使用固定长度策略把 Document 列表切分为多个 Chunk Document。

    实现方式：校验切块参数后创建 LangChain 的
    RecursiveCharacterTextSplitter，再逐个 Document 调用其 split_documents()
    完成切分。LangChain 将原始 Document 的 Metadata 复制到每个 Chunk，业务代码
    再补充 chunk_id。ID 使用文档 ID、页码和从 01 开始的 Chunk 顺序组成；没有
    document_id 时使用来源文件名，没有来源时使用输入顺序生成回退 ID。

    参数：
        documents: 待切分的 LangChain Document 列表；每个元素应包含文本内容。
        chunk_size: 单个 Chunk 的最大字符数；必须为正整数，默认 800。
        chunk_overlap: 相邻 Chunk 的重叠字符数；必须为非负整数且小于 chunk_size，
            默认 150。

    返回：
        切分后的 Chunk Document 列表；每个 Chunk 都保留原始 Metadata，并额外
        包含 chunk_id，输入为空列表时返回空列表。

    异常：
        ValueError: chunk_size 或 chunk_overlap 不符合约束，或 page Metadata
            不是整数时抛出。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for document_index, document in enumerate(documents):
        document_chunks = text_splitter.split_documents([document])
        document_id = document.metadata.get("document_id")
        if not document_id:
            source = document.metadata.get("source")
            document_id = Path(str(source)).stem if source else f"document_{document_index + 1}"

        page = document.metadata.get("page")
        page_number = int(page) + 1 if page is not None else document_index + 1
        for chunk_index, chunk in enumerate(document_chunks, start=1):
            chunk.metadata["chunk_id"] = (
                f"{document_id}_p{page_number}_chunk_{chunk_index:02d}"
            )
            chunks.append(chunk)
    return chunks
