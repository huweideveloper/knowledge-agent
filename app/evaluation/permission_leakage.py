"""阶段 14.7：评测权限越权导致的检索和 Context 数据泄露。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.retrieval.vector_retriever import RetrievedChunk
from app.security import User


# 增加于阶段 14.7：定义单条权限泄露评测样本。
@dataclass(frozen=True)
class PermissionEvaluationCase:
    """保存用户、越权问题、受限数据标记和实际输出链路。"""

    question: str
    user: User
    restricted_document_ids: frozenset[str]
    restricted_markers: frozenset[str]
    retrieved_results: tuple[RetrievedChunk, ...]
    context: str


# 增加于阶段 14.7：定义权限泄露评测汇总报告。
@dataclass(frozen=True)
class PermissionLeakageReport:
    """保存权限样本数、泄露数量、泄露对象和泄露率。"""

    total_cases: int
    total_retrieved_chunks: int
    leaked_cases: int
    leaked_chunks: int
    leaked_contexts: int
    leaked_case_questions: tuple[str, ...]
    leaked_documents: tuple[str, ...]
    leakage_rate: float


# 增加于阶段 14.7：根据实际 Retrieval 和 Context 输出计算权限泄露指标。
def evaluate_permission_leakage(
    cases: Iterable[PermissionEvaluationCase],
) -> PermissionLeakageReport:
    """计算权限评测样本中受限数据进入结果或 Context 的比例。

    实现方式：校验每条 PermissionEvaluationCase，遍历实际 RetrievedChunk 的文档 ID
    和正文，检查受限文档 ID/敏感标记是否出现；同时扫描最终 Context，捕获绕过结果
    列表直接进入模型输入的泄露。任一通道出现受限数据即计为一个泄露案例，最后返回
    泄露案例、泄露 Chunk、泄露 Context、泄露文档和泄露率。该评测不修改检索结果，
    只观察权限过滤后的真实输出。

    参数：
        cases: 权限评测样本可迭代对象；每项必须包含用户、受限数据标记、检索结果和
            最终 Context，至少一条样本。

    返回：
        PermissionLeakageReport：包含总样本数、检索 Chunk 总数、泄露案例数、泄露
            Chunk/Context 数、泄露问题、泄露文档和 0 到 1 之间的泄露率。

    异常：
        TypeError：cases 不可迭代、样本/用户/Chunk/Metadata 类型不正确时抛出。
        ValueError：cases 为空、问题为空、受限标记为空、Context 非字符串或 Chunk
            正文为空时抛出。
    """
    if isinstance(cases, (str, bytes)):
        raise TypeError("cases must be an iterable of PermissionEvaluationCase")
    try:
        case_list = list(cases)
    except TypeError as error:
        raise TypeError("cases must be an iterable of PermissionEvaluationCase") from error
    if not case_list:
        raise ValueError("cases must not be empty")

    leaked_case_questions: list[str] = []
    leaked_documents: list[str] = []
    seen_leaked_documents: set[str] = set()
    total_retrieved_chunks = 0
    leaked_chunks = 0
    leaked_contexts = 0
    for case in case_list:
        if not isinstance(case, PermissionEvaluationCase):
            raise TypeError("cases must contain only PermissionEvaluationCase objects")
        if not isinstance(case.question, str) or not case.question.strip():
            raise ValueError("PermissionEvaluationCase.question must not be empty")
        if not isinstance(case.user, User):
            raise TypeError("PermissionEvaluationCase.user must be a User")
        if not isinstance(case.restricted_document_ids, (set, frozenset)):
            raise TypeError("restricted_document_ids must be a set of strings")
        if not isinstance(case.restricted_markers, (set, frozenset)):
            raise TypeError("restricted_markers must be a set of strings")
        restricted_documents = {
            document_id.strip()
            for document_id in case.restricted_document_ids
            if isinstance(document_id, str) and document_id.strip()
        }
        restricted_markers = {
            marker.strip()
            for marker in case.restricted_markers
            if isinstance(marker, str) and marker.strip()
        }
        if not restricted_documents and not restricted_markers:
            raise ValueError("at least one restricted document or marker is required")
        if len(restricted_documents) != len(case.restricted_document_ids):
            raise ValueError("restricted_document_ids must contain non-empty strings")
        if len(restricted_markers) != len(case.restricted_markers):
            raise ValueError("restricted_markers must contain non-empty strings")
        if not isinstance(case.context, str):
            raise TypeError("PermissionEvaluationCase.context must be a string")
        if isinstance(case.retrieved_results, (str, bytes)):
            raise TypeError("retrieved_results must be an iterable of RetrievedChunk")
        try:
            result_list = list(case.retrieved_results)
        except TypeError as error:
            raise TypeError(
                "retrieved_results must be an iterable of RetrievedChunk"
            ) from error

        case_leaked = False
        total_retrieved_chunks += len(result_list)
        for chunk in result_list:
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("retrieved_results must contain only RetrievedChunk objects")
            if not isinstance(chunk.metadata, Mapping):
                raise ValueError("RetrievedChunk.metadata must be an object")
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise ValueError("RetrievedChunk.content must not be empty")
            document_id = chunk.metadata.get("document_id") or chunk.source
            document_leaked = document_id in restricted_documents
            marker_leaked = any(marker in chunk.content for marker in restricted_markers)
            if document_leaked or marker_leaked:
                case_leaked = True
                leaked_chunks += 1
                if isinstance(document_id, str) and document_id in restricted_documents:
                    if document_id not in seen_leaked_documents:
                        seen_leaked_documents.add(document_id)
                        leaked_documents.append(document_id)

        context_leaked = any(
            marker in case.context for marker in restricted_markers
        ) or any(document_id in case.context for document_id in restricted_documents)
        if context_leaked:
            case_leaked = True
            leaked_contexts += 1
        if case_leaked:
            leaked_case_questions.append(case.question)

    total_cases = len(case_list)
    leaked_cases = len(leaked_case_questions)
    return PermissionLeakageReport(
        total_cases=total_cases,
        total_retrieved_chunks=total_retrieved_chunks,
        leaked_cases=leaked_cases,
        leaked_chunks=leaked_chunks,
        leaked_contexts=leaked_contexts,
        leaked_case_questions=tuple(leaked_case_questions),
        leaked_documents=tuple(leaked_documents),
        leakage_rate=leaked_cases / total_cases,
    )
