"""阶段 13.1：演示最低相关度阈值如何阻止低置信度回答。"""

import sys
from collections.abc import Callable
from pathlib import Path


# 增加于阶段 13.1：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pipeline import MIN_RELEVANCE_SCORE, RagPipeline
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 13.1：提供低相关度检索结果，演示阈值拒答分支。
def _low_confidence_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """返回最高相关度为 0.49 的离线测试结果。

    实现方式：校验 Pipeline 的 Query 和 Top K 后构造一条完整 RetrievedChunk，避免
    Demo 依赖 Qdrant 或 Embedding 服务，同时让阈值判断结果稳定可复现。

    参数：
        query: Pipeline 传入的问题，必须是 Demo 展示的无答案问题。
        top_k: Pipeline 请求的候选数量，必须为正整数且本 Demo 使用 5。

    返回：
        list[RetrievedChunk]：包含一条 score 为 0.49 的检索结果。

    异常：
        AssertionError：Pipeline 入参不符合 Demo 预期时抛出。
    """
    assert query == "公司是否报销私人旅行？"
    assert top_k == 5
    return [
        RetrievedChunk(
            content="普通员工住宿标准为 600 元/晚。",
            score=0.49,
            source="travel_policy_2026",
            page=1,
            metadata={"source": "travel_policy_2026", "page": 1},
        )
    ]


# 增加于阶段 13.1：记录模型调用次数，证明拒答分支不会进入 LLM。
def _recording_model(calls: list[list[object]]) -> Callable[[list[object]], str]:
    """创建一个记录调用次数并返回固定回答的离线模型。

    实现方式：每次模型收到 Prompt 时追加一条记录并返回固定文本；本 Demo 预期
    阈值过滤后模型不会被调用，因此 calls 应保持为空。

    参数：
        calls: 用于记录模型调用消息的可变列表。

    返回：
        callable：符合 RagPipeline LLM 调用约定的测试模型。

    异常：
        无主动抛出的异常。
    """
    # 增加于阶段 13.1：实现 Demo 模型的单次调用记录。
    def model(messages: list[object]) -> str:
        """记录一次模型调用并返回固定文本。

        实现方式：保存收到的消息列表后返回固定回答，不访问外部模型服务。

        参数：
            messages: RagPipeline 生成的 Prompt 消息列表。

        返回：
            str：固定的测试回答。

        异常：
            无主动抛出的异常。
        """
        calls.append(messages)
        return "不应在本 Demo 的低相关度分支中出现。"

    return model


# 增加于阶段 13.1：执行阈值过滤并打印可直接观察的输入、处理结果和输出。
# 修改于阶段 13.2：使用统一的固定拒答话术完成阈值拒答验收。
def run_demo() -> None:
    """演示最高结果低于阈值时 Pipeline 直接拒答。

    实现方式：注入固定低相关度 Retriever 和可记录调用的模型，执行正式
    RagPipeline；检查 Chunk、Context、模型调用和回答均符合拒答分支后打印验收信息。

    参数：
        无入参；使用固定问题和 score 为 0.49 的测试资料。

    返回：
        无返回值；Demo 结果通过标准输出打印。

    异常：
        RuntimeError：低相关度资料被保留、Context 非空、模型被调用或回答不符合
            既有拒答结果时抛出。
    """
    question = "公司是否报销私人旅行？"
    input_score = 0.49
    calls: list[list[object]] = []
    result = RagPipeline(
        retriever=_low_confidence_retriever,
        llm=_recording_model(calls),
    ).invoke(question)
    if result.chunks or result.context or calls:
        raise RuntimeError("无答案处理验收失败：低相关度结果进入了回答流程")
    if result.answer != "根据当前知识库，没有找到足够信息回答该问题。":
        raise RuntimeError("无答案处理验收失败：拒答结果不符合预期")

    print("=== 阶段 13.1 最低相关度阈值 Demo ===")
    print(f"输入问题: {question}")
    print(f"输入最高相关度: {input_score:.2f}")
    print(f"最低相关度阈值: {MIN_RELEVANCE_SCORE:.2f}")
    print("关键处理结果: 低于阈值的检索结果已在 Context 前过滤")
    print(f"过滤后 Chunk 数量: {len(result.chunks)}")
    print(f"Context 状态: {'空' if not result.context else '非空'}")
    print(f"模型调用次数: {len(calls)}")
    print(f"最终输出: {result.answer}")
    print("无答案处理验收: 通过")


# 增加于阶段 13.1：提供最低相关度阈值 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
