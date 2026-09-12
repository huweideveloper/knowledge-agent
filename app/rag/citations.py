"""阶段 10.2：将结构化 LLM 输出映射为可信 Citation。"""

from collections.abc import Iterable, Mapping
import json
import re

from pydantic import BaseModel, Field

from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 10.2：定义 LLM 必须返回的正文和 Chunk ID 结构。
class StructuredAnswer(BaseModel):
    """表示 LLM 返回的回答正文和其声明使用的 Chunk ID 列表。"""

    answer: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


# 增加于阶段 10.2：定义程序根据真实 Chunk 生成的可信引用对象。
class Citation(BaseModel):
    """表示一条由真实检索 Metadata 映射出的 Citation。"""

    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int | str
    label: str = Field(min_length=1)


# 增加于阶段 10.2：定义保留回答和结构化引用的最终输出。
class CitedAnswer(BaseModel):
    """表示回答正文及其可追溯的 Citation 列表。"""

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


# 增加于阶段 10.3：定义 Citation 内容支持验证的结构化结果。
class CitationVerification(BaseModel):
    """记录引用是否支持答案及匹配、缺失的事实。"""

    supported: bool
    cited_chunk_ids: list[str]
    matched_facts: list[str]
    missing_facts: list[str]


# 增加于阶段 10.3：归一化答案和 Chunk 文本中的空白字符。
def _normalize_citation_text(value: str) -> str:
    """去除文本空白，便于比较 PDF 换行造成的数字事实差异。

    实现方式：确认输入为字符串后删除所有 Unicode 空白；不做同义词替换或
    模糊改写，避免验证器把不相同的数字事实误判为相同。

    参数：
        value: 待比较的答案或 Chunk 正文。

    返回：
        str：去除空白后的文本。

    异常：
        TypeError: value 不是字符串时抛出。
    """
    if not isinstance(value, str):
        raise TypeError("citation text must be a string")
    return re.sub(r"\s+", "", value)


# 增加于阶段 10.3：提取答案中需要从资料核验的数字事实。
def _extract_numeric_facts(answer: str) -> list[str]:
    """提取带常见业务单位的数字事实并统一其空白格式。

    实现方式：识别金额/晚、金额/天、小时、天、次、公里和百分比等事实，将
    “600 元/晚”和“600元/晚”归一为同一比较值；只返回答案实际出现的数字事实。

    参数：
        answer: LLM 输出的答案正文，必须是字符串。

    返回：
        list[str]：按出现顺序去重后的数字事实列表；没有可识别事实时返回空列表。

    异常：
        TypeError: answer 不是字符串时抛出。
    """
    normalized_answer = _normalize_citation_text(answer)
    matches = re.findall(
        r"\d+(?:\.\d+)?(?:元/(?:晚|天)|小时|天|次|公里|%)",
        normalized_answer,
    )
    return list(dict.fromkeys(matches))


# 增加于阶段 10.3：验证引用 Chunk 正文是否支持答案事实。
def verify_citation_support(
    cited_answer: CitedAnswer,
    chunks: Iterable[RetrievedChunk],
) -> CitationVerification:
    """验证 CitedAnswer 的引用正文是否真的包含答案中的支持事实。

    实现方式：根据本次检索结果建立 chunk_id 到 RetrievedChunk 的映射，仅收集
    CitedAnswer 声明的 Chunk 正文；提取答案中的数字事实并逐项检查是否存在于这些
    正文中。验证只使用程序保存的 Chunk 内容，不信任 Citation 自带的 source/page。

    参数：
        cited_answer: 已通过 10.2 映射的结构化回答和 Citation。
        chunks: 本次检索得到的 RetrievedChunk 可迭代对象。

    返回：
        CitationVerification：包含 supported、引用 ID、匹配事实和缺失事实。

    异常：
        TypeError: cited_answer 类型错误、chunks 不可迭代或元素类型错误时抛出。
        ValueError: Chunk 缺少 chunk_id，或 Citation ID 不在本次检索结果中时抛出。
    """
    if not isinstance(cited_answer, CitedAnswer):
        raise TypeError("cited_answer must be a CitedAnswer")
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of RetrievedChunk")
    try:
        chunk_list = list(chunks)
    except TypeError as error:
        raise TypeError("chunks must be an iterable of RetrievedChunk") from error

    chunk_map: dict[str, RetrievedChunk] = {}
    for chunk in chunk_list:
        if not isinstance(chunk, RetrievedChunk):
            raise TypeError("chunks must contain only RetrievedChunk objects")
        chunk_id = chunk.chunk_id
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            metadata_chunk_id = chunk.metadata.get("chunk_id")
            chunk_id = (
                metadata_chunk_id.strip()
                if isinstance(metadata_chunk_id, str) and metadata_chunk_id.strip()
                else None
            )
        if not chunk_id:
            raise ValueError("each chunk must have a non-empty chunk_id")
        if chunk_id in chunk_map:
            raise ValueError(f"duplicate chunk_id in chunk map: {chunk_id}")
        chunk_map[chunk_id] = chunk

    cited_chunk_ids = [citation.chunk_id for citation in cited_answer.citations]
    cited_contents: list[str] = []
    for chunk_id in cited_chunk_ids:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            raise ValueError(f"citation ID not found in retrieved chunks: {chunk_id}")
        cited_contents.append(chunk.content)

    normalized_cited_content = _normalize_citation_text("\n".join(cited_contents))
    numeric_facts = _extract_numeric_facts(cited_answer.answer)
    matched_facts = [fact for fact in numeric_facts if fact in normalized_cited_content]
    missing_facts = [fact for fact in numeric_facts if fact not in normalized_cited_content]
    supported = bool(numeric_facts) and not missing_facts
    return CitationVerification(
        supported=supported,
        cited_chunk_ids=cited_chunk_ids,
        matched_facts=matched_facts,
        missing_facts=missing_facts,
    )


# 增加于阶段 10.2：解析来自 LLM 的 Pydantic、字典或 JSON 字符串输出。
def parse_structured_answer(
    value: StructuredAnswer | Mapping[str, object] | str,
) -> StructuredAnswer:
    """将多种 LLM 结构化输出形式统一校验为 StructuredAnswer。

    实现方式：已是 Pydantic 对象时直接复用；字典交给 model_validate；字符串先
    解析 JSON，再交给 Pydantic 校验。函数只负责结构校验，不接受 LLM 直接提供的
    source、page 或文件名字段。

    参数：
        value: LLM 返回的 StructuredAnswer、字段字典或 JSON 字符串。

    返回：
        StructuredAnswer：包含非空 answer 和至少一个 citation Chunk ID 的对象。

    异常：
        TypeError: value 不是支持的输出类型时抛出。
        ValueError: value 是非法 JSON，或 JSON 顶层不是对象时抛出。
        pydantic.ValidationError: 输出缺少必填字段或字段类型不符合约束时抛出。
    """
    if isinstance(value, StructuredAnswer):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("LLM structured output 不是合法 JSON") from error
        if not isinstance(decoded, Mapping):
            raise ValueError("LLM structured output 顶层必须是对象")
        return StructuredAnswer.model_validate(decoded)
    if isinstance(value, Mapping):
        return StructuredAnswer.model_validate(value)
    raise TypeError("value must be StructuredAnswer, mapping, or JSON string")


# 增加于阶段 10.2：根据真实 Chunk 来源生成固定展示标签。
def _format_citation_label(source: str, page: int | str) -> str:
    """把真实来源和页码格式化为用户可读的 Citation 标签。

    实现方式：使用程序从检索对象取得的 source 和 page，按《来源》P页码格式拼接；
    不接受来自 LLM 的文件名或页码，因此展示值只能来自 Chunk Metadata。

    参数：
        source: 检索结果来源文件或业务文档标识，必须为非空字符串。
        page: 检索结果页码，可以是整数或非空字符串。

    返回：
        str：例如《差旅管理制度.pdf》P12 的来源标签。

    异常：
        ValueError: source 为空、page 为空或 page 类型不符合约束时抛出。
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("citation source must be a non-empty string")
    if isinstance(page, bool) or not isinstance(page, (int, str)):
        raise ValueError("citation page must be an integer or non-empty string")
    if isinstance(page, str) and not page.strip():
        raise ValueError("citation page must not be empty")
    return f"《{source.strip()}》P{page}"


# 增加于阶段 10.2：把 LLM 声明的 Chunk ID 映射为真实 Citation。
def build_cited_answer(
    value: StructuredAnswer | Mapping[str, object] | str,
    chunks: Iterable[RetrievedChunk],
) -> CitedAnswer:
    """根据真实检索 Chunk Map 生成可信的回答和 Citation。

    实现方式：先用 Pydantic 校验 LLM 的 answer/citations 结构，再建立
    `chunk_id -> RetrievedChunk` 索引；每个 LLM 声明的 ID 必须存在于该索引，随后
    从对应 Chunk 的 source/page 生成 Citation，完全忽略模型可能编造的来源字段。

    参数：
        value: LLM 返回的 StructuredAnswer、字段字典或 JSON 字符串。
        chunks: 本次检索得到的 RetrievedChunk 可迭代对象，每项必须有非空
            chunk_id、source 和 page。

    返回：
        CitedAnswer：保留原回答正文，并包含按 LLM ID 顺序映射出的可信 Citation。

    异常：
        TypeError: chunks 不可迭代、元素类型错误或输入类型不支持时抛出。
        ValueError: Chunk 来源不完整、Chunk ID 重复、或 LLM 引用了不存在的 ID 时抛出。
        pydantic.ValidationError: LLM 结构化输出不符合模型时抛出。
    """
    structured_answer = parse_structured_answer(value)
    if not structured_answer.answer.strip():
        raise ValueError("structured answer must not be empty")
    if isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be an iterable of RetrievedChunk")
    try:
        chunk_list = list(chunks)
    except TypeError as error:
        raise TypeError("chunks must be an iterable of RetrievedChunk") from error

    chunk_map: dict[str, RetrievedChunk] = {}
    for chunk in chunk_list:
        if not isinstance(chunk, RetrievedChunk):
            raise TypeError("chunks must contain only RetrievedChunk objects")
        chunk_id = chunk.chunk_id
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            metadata_chunk_id = chunk.metadata.get("chunk_id")
            chunk_id = (
                metadata_chunk_id.strip()
                if isinstance(metadata_chunk_id, str) and metadata_chunk_id.strip()
                else None
            )
        if not chunk_id:
            raise ValueError("each chunk must have a non-empty chunk_id")
        if chunk_id in chunk_map:
            raise ValueError(f"duplicate chunk_id in chunk map: {chunk_id}")
        if not isinstance(chunk.source, str) or not chunk.source.strip():
            raise ValueError(f"chunk {chunk_id} has no source")
        if chunk.page is None or isinstance(chunk.page, bool):
            raise ValueError(f"chunk {chunk_id} has no page")
        chunk_map[chunk_id] = chunk

    citations: list[Citation] = []
    seen_ids: set[str] = set()
    for citation_id in structured_answer.citations:
        if not isinstance(citation_id, str) or not citation_id.strip():
            raise ValueError("citation IDs must be non-empty strings")
        citation_id = citation_id.strip()
        if citation_id in seen_ids:
            raise ValueError(f"duplicate citation ID: {citation_id}")
        chunk = chunk_map.get(citation_id)
        if chunk is None:
            raise ValueError(f"citation ID not found in retrieved chunks: {citation_id}")
        seen_ids.add(citation_id)
        citations.append(
            Citation(
                chunk_id=citation_id,
                source=chunk.source.strip(),
                page=chunk.page,
                label=_format_citation_label(chunk.source, chunk.page),
            )
        )

    return CitedAnswer(answer=structured_answer.answer.strip(), citations=citations)
