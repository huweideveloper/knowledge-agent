"""阶段 9.1–9.2：构造并控制送入模型的检索 Context。"""

from collections.abc import Iterable

from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 9.2：定义 V1 送入模型的最大 Chunk 数量。
MAX_CONTEXT_CHUNKS = 5


# 增加于阶段 9.1：把多个标准检索结果格式化为模型可读的资料 Context。
def build_context(chunks: Iterable[RetrievedChunk]) -> str:
    """把多个 RetrievedChunk 按固定资料块格式拼接成 Context。

    实现方式：先消费输入迭代器，逐项校验对象和正文类型，再按输入顺序生成
    `[资料N]`、来源和正文三个字段；资料块之间使用一个空行分隔，不做重排、
    去重或字符级截断。来源缺失时使用“未知来源”。

    参数：
        chunks: 待拼接的 RetrievedChunk 可迭代对象；允许为空，空输入返回空字符串。

    返回：
        str：按固定格式拼接后的 Context；输入为空时返回空字符串。

    异常：
        TypeError: chunks 不可迭代、元素不是 RetrievedChunk，或正文不是字符串时抛出。
    """
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of RetrievedChunk")

    try:
        chunk_list = list(chunks)
    except TypeError as error:
        raise TypeError("chunks must be an iterable of RetrievedChunk") from error

    blocks: list[str] = []
    for index, chunk in enumerate(chunk_list, start=1):
        if not isinstance(chunk, RetrievedChunk):
            raise TypeError("chunks must contain only RetrievedChunk objects")
        if not isinstance(chunk.content, str):
            raise TypeError("RetrievedChunk.content must be a string")
        source = chunk.source.strip() if isinstance(chunk.source, str) else "未知来源"
        if not source:
            source = "未知来源"
        blocks.append(
            f"[资料{index}]\n来源：\n{source}\n正文：\n{chunk.content.strip()}"
        )

    return "\n\n".join(blocks)


# 增加于阶段 9.2：限制进入 Context 的候选 Chunk 数量。
def limit_chunks(
    chunks: Iterable[RetrievedChunk],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
) -> list[RetrievedChunk]:
    """按原检索顺序保留最多指定数量的 RetrievedChunk。

    实现方式：将输入迭代器物化一次，验证数量上限和每个元素的标准对象类型，
    然后返回前 max_chunks 项；函数不修改原对象、不重排、不去重，也不截断正文。

    参数：
        chunks: 按相关度排序的候选 RetrievedChunk 可迭代对象。
        max_chunks: 最多保留的 Chunk 数量，必须为正整数，默认 5。

    返回：
        list[RetrievedChunk]：按输入顺序保留的前 max_chunks 项；候选不足时返回全部。

    异常：
        TypeError: max_chunks 不是整数，或 chunks 元素不是 RetrievedChunk 时抛出。
        ValueError: max_chunks 小于等于 0 时抛出。
    """
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int):
        raise TypeError("max_chunks must be an integer")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero")
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of RetrievedChunk")

    try:
        chunk_list = list(chunks)
    except TypeError as error:
        raise TypeError("chunks must be an iterable of RetrievedChunk") from error
    if any(not isinstance(chunk, RetrievedChunk) for chunk in chunk_list):
        raise TypeError("chunks must contain only RetrievedChunk objects")
    return chunk_list[:max_chunks]
