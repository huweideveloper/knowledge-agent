"""阶段 6 Embedding 能力的模型选择与接口封装。"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math

from langchain_core.embeddings import Embeddings


# 增加于阶段 6.1：定义 Embedding 模型选择的数据结构。
@dataclass(frozen=True)
class EmbeddingModelSelection:
    """描述当前 V1 使用的独立 Embedding 模型。"""

    provider: str
    model_name: str
    runtime: str
    model_type: str
    purpose: str
    selection_reason: str


# 增加于阶段 6.1：返回当前 V1 的 Embedding 模型选择。
# 修改于阶段 6.2：改用 FastEmbed 运行时以支持当前 CPU 环境。
def get_embedding_model_selection() -> EmbeddingModelSelection:
    """返回当前 V1 选定的独立 Embedding 模型及其选择理由。

    实现方式：返回一个不可变的模型选择配置，使用 Hugging Face Hub 上的
    BAAI/bge-small-zh-v1.5 作为模型名称，并指定 FastEmbed 作为本地运行时。
    本函数只描述模型选择，不下载模型、不调用网络，也不执行文本向量化。

    参数：
        无入参。

    返回：
        EmbeddingModelSelection：包含 provider、模型名称、运行时、模型职责、
        适用目的和选择理由的不可变配置对象。

    异常：
        无主动抛出的异常。
    """
    return EmbeddingModelSelection(
        provider="huggingface",
        model_name="BAAI/bge-small-zh-v1.5",
        runtime="fastembed",
        model_type="embedding",
        purpose="中文企业文档与用户 Query 的语义表示",
        selection_reason=(
            "面向中文语义检索，轻量且支持本地 CPU 运行，并与 DeepSeek 聊天模型解耦"
        ),
    )


# 增加于阶段 6.2：懒加载并缓存 LangChain Embeddings 实例。
@lru_cache(maxsize=1)
def _get_embedding_model() -> Embeddings:
    """创建并缓存阶段 6.1 选定的 LangChain Embeddings 实例。

    实现方式：读取正式模型选择配置，按 FastEmbed 运行时创建 LangChain 的
    FastEmbedEmbeddings；模型只在第一次真正向量化时加载，后续文档和 Query
    调用复用同一实例，避免重复初始化模型。本函数不负责维度验收。

    参数：
        无入参。

    返回：
        Embeddings：实现 LangChain Embeddings 接口的本地模型实例。

    异常：
        RuntimeError: 未安装 fastembed，或当前模型选择不是 FastEmbed 运行时。
    """
    selection = get_embedding_model_selection()
    if selection.runtime != "fastembed":
        raise RuntimeError(
            f"不支持的 Embedding 运行时：{selection.runtime}"
        )

    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Embedding 依赖未安装，请执行 pip install -r requirements.txt"
        ) from error

    return FastEmbedEmbeddings(model_name=selection.model_name)


# 增加于阶段 6.2：封装批量文档向量化接口。
def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """将多个文档文本转换为对应的 Embedding 向量。

    实现方式：校验输入为字符串序列，空序列直接返回空列表；非空输入委托给
    缓存的 LangChain Embeddings 实例，由 BAAI/bge-small-zh-v1.5 生成一个与每个
    文本一一对应的浮点向量。业务代码只依赖本函数，不直接绑定底层模型供应商。

    参数：
        texts: 待向量化的文档文本序列；每个元素必须是字符串，可以为空序列。

    返回：
        List[List[float]]：与输入文本数量相同的向量列表；输入为空时返回空列表。

    异常：
        TypeError: texts 是字符串本身，或序列中包含非字符串元素。
        RuntimeError: Embedding 运行时依赖未安装或模型初始化失败。
    """
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")

    text_list = list(texts)
    if any(not isinstance(text, str) for text in text_list):
        raise TypeError("texts must contain only strings")
    if not text_list:
        return []

    return _get_embedding_model().embed_documents(text_list)


# 增加于阶段 6.2：封装单个用户 Query 向量化接口。
def embed_query(query: str) -> list[float]:
    """将一个用户 Query 转换为 Embedding 向量。

    实现方式：校验 Query 为字符串后，委托给与文档向量化相同的缓存模型实例，
    保证文档和 Query 使用同一模型与同一向量空间；本函数不执行检索。

    参数：
        query: 待向量化的用户 Query，必须是字符串；空字符串仍交由模型处理，
            由后续任务决定是否禁止空 Query。

    返回：
        List[float]：代表该 Query 语义的浮点向量。

    异常：
        TypeError: query 不是字符串。
        RuntimeError: Embedding 运行时依赖未安装或模型初始化失败。
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    return _get_embedding_model().embed_query(query)


# 增加于阶段 6.3：验证文档向量和 Query 向量使用一致的维度。
def validate_embedding_dimensions(
    document_vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
) -> int:
    """验证文档向量和 Query 向量的维度一致，并返回共同维度。

    实现方式：先检查文档向量和 Query 向量都不为空，再确认所有文档向量彼此
    具有相同长度，最后将该长度与 Query 向量长度比较。校验在向量写入 Qdrant
    前执行，避免维度错误延迟到向量数据库阶段才暴露。

    参数：
        document_vectors: 文档 Embedding 向量序列；至少包含一个非空向量。
        query_vector: 用户 Query 的 Embedding 向量；必须是非空向量。

    返回：
        int：文档和 Query 共用的向量维度。

    异常：
        ValueError: 任一输入为空、文档向量之间维度不一致，或 Query 与文档
            向量维度不一致时抛出。
    """
    if not document_vectors:
        raise ValueError("document_vectors must not be empty")
    if not query_vector:
        raise ValueError("query_vector must not be empty")

    document_dimensions = {len(vector) for vector in document_vectors}
    if len(document_dimensions) != 1:
        raise ValueError("document vectors must have the same dimension")

    document_dimension = document_dimensions.pop()
    if document_dimension <= 0:
        raise ValueError("embedding dimension must be greater than zero")
    if len(query_vector) != document_dimension:
        raise ValueError(
            "query vector dimension must match document vector dimension"
        )

    return document_dimension


# 增加于阶段 6.4：使用标准库计算两个 Embedding 向量的余弦相似度。
def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """计算两个 Embedding 向量的余弦相似度。

    实现方式：先校验两个向量非空且维度一致，再使用标准库 math 计算点积和
    两个向量的欧氏范数，返回点积除以范数乘积的结果。函数只负责相似度计算，
    不负责文本向量化、持久化或排序。

    参数：
        vector_a: 第一个 Embedding 向量；必须是包含数值的非空序列。
        vector_b: 第二个 Embedding 向量；必须与 vector_a 维度相同且非空。

    返回：
        float：两个向量的余弦相似度，通常位于 -1 到 1 之间。

    异常：
        TypeError: 任一参数是字符串或向量元素不是可计算数值时抛出。
        ValueError: 任一向量为空、维度不一致或向量范数为零时抛出。
    """
    if isinstance(vector_a, str) or isinstance(vector_b, str):
        raise TypeError("vectors must be numeric sequences, not strings")
    if not vector_a or not vector_b:
        raise ValueError("vectors must not be empty")
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same dimension")

    try:
        dot_product = sum(
            component_a * component_b
            for component_a, component_b in zip(vector_a, vector_b)
        )
        norm_a = math.sqrt(sum(component * component for component in vector_a))
        norm_b = math.sqrt(sum(component * component for component in vector_b))
    except TypeError as error:
        raise TypeError("vectors must contain numeric values") from error

    if norm_a == 0 or norm_b == 0:
        raise ValueError("zero vectors do not have a cosine similarity")

    return dot_product / (norm_a * norm_b)
