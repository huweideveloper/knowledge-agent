"""阶段 13.2：演示 Prompt 规则如何让资料不足时输出固定拒答话术。"""

import sys
from pathlib import Path


# 增加于阶段 13.2：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.llm import GroundedFallbackChatModel
from app.rag.pipeline import RagPipeline
from app.rag.prompt import ABSTAIN_ANSWER, build_chat_prompt
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 13.2：定义不包含问题答案的固定输入问题和 Context。
QUESTION = "公司是否报销私人旅行？"
CONTEXT = "员工应在每年一季度完成安全培训。"


# 增加于阶段 13.2：提供能进入模型但无法回答当前问题的固定检索结果。
def _insufficient_context_retriever(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """返回相关度达标但不包含问题答案的检索结果。

    实现方式：校验 Pipeline 传入的问题和候选数量，返回一条 score 为 0.6 的安全
    培训资料，让阶段 13.1 放行该 Context，再观察阶段 13.2 的 Prompt 拒答话术。

    参数：
        query: Pipeline 传入的问题，必须等于 Demo 的私人旅行报销问题。
        top_k: Pipeline 请求的候选数量，必须为 5。

    返回：
        list[RetrievedChunk]：包含一条与当前问题不匹配的完整检索结果。

    异常：
        AssertionError：Pipeline 入参不符合 Demo 预期时抛出。
    """
    assert query == QUESTION
    assert top_k == 5
    return [
        RetrievedChunk(
            content=CONTEXT,
            score=0.6,
            source="safety_policy_2026",
            page=2,
            metadata={"source": "safety_policy_2026", "page": 2},
        )
    ]


# 增加于阶段 13.2：渲染 Prompt、执行 Pipeline 并打印模型可见规则和最终输出。
def run_demo() -> None:
    """演示非空但不足的 Context 如何触发固定拒答话术。

    实现方式：先使用正式 build_chat_prompt() 检查固定拒答规则，再用本地
    GroundedFallbackChatModel 执行完整 RagPipeline；该模型不访问网络，找不到住宿
    答案时遵循共享 ABSTAIN_ANSWER，最后打印输入、关键处理结果和最终输出。

    参数：
        无入参；使用预设的私人旅行报销问题和安全培训资料。

    返回：
        无返回值；Prompt 规则、Context 状态、模型回答和验收结论通过标准输出打印。

    异常：
        RuntimeError：拒答规则没有进入 System Prompt、Context 没有进入模型，或
            最终输出不是需求文档规定的话术时抛出。
    """
    prompt_value = build_chat_prompt(QUESTION, CONTEXT)
    system_message = prompt_value.to_messages()[0]
    if ABSTAIN_ANSWER not in system_message.content:
        raise RuntimeError("Prompt 拒答验收失败：固定拒答规则未注入 System Prompt")

    result = RagPipeline(
        retriever=_insufficient_context_retriever,
        llm=GroundedFallbackChatModel(),
    ).invoke(QUESTION)
    if len(result.chunks) != 1 or CONTEXT not in result.context:
        raise RuntimeError("Prompt 拒答验收失败：输入 Context 未进入模型")
    if result.answer != ABSTAIN_ANSWER:
        raise RuntimeError("Prompt 拒答验收失败：模型没有输出固定拒答话术")

    print("=== 阶段 13.2 Prompt 拒答规则 Demo ===")
    print(f"输入问题: {QUESTION}")
    print("输入 Context: 非空但不包含问题答案")
    print("关键处理结果: Prompt 已要求资料不足时只输出固定拒答话术")
    print("Prompt 拒答规则: 已注入")
    print(f"Context 进入模型: {'是' if result.context else '否'}")
    print(f"最终输出: {result.answer}")
    print("Prompt 拒答验收: 通过")


# 增加于阶段 13.2：提供 Prompt 拒答规则 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
