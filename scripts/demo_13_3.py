"""阶段 13.3：演示无答案问题的 False Answer Rate 评测。"""

import sys
from pathlib import Path


# 增加于阶段 13.3：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.false_answer_rate import EvaluationCase, evaluate_false_answer_rate
from app.rag.llm import GroundedFallbackChatModel
from app.rag.pipeline import RagPipeline
from app.retrieval.vector_retriever import RetrievedChunk


# 增加于阶段 13.3：定义可回答和无答案评测问题。
EVALUATION_INPUTS = (
    ("公司是否报销私人旅行？", True),
    ("公司是否报销宠物旅行？", True),
    ("普通员工住宿标准是多少？", False),
)


# 增加于阶段 13.3：提供固定评测资料，覆盖拒答和正常回答两类输入。
def _evaluation_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """根据评测问题返回确定性的本地检索结果。

    实现方式：校验 Pipeline 的候选数量；两个无答案问题返回安全培训资料，正常
    住宿问题返回包含明确金额的资料，所有结果分数均高于阶段 13.1 阈值，使评测
    重点落在模型是否正确拒答而不是检索是否被阈值拦截。

    参数：
        query: Pipeline 传入的评测问题，必须属于 EVALUATION_INPUTS。
        top_k: Pipeline 请求的候选数量，必须为 5。

    返回：
        list[RetrievedChunk]：与问题类型匹配的一条固定检索结果。

    异常：
        AssertionError：Pipeline 传入未知问题或错误候选数量时抛出。
    """
    assert query in {question for question, _ in EVALUATION_INPUTS}
    assert top_k == 5
    if query == "普通员工住宿标准是多少？":
        return [
            RetrievedChunk(
                content="普通员工 600 元/晚。",
                score=0.95,
                source="travel_policy_2026",
                page=1,
                metadata={"source": "travel_policy_2026", "page": 1},
            )
        ]
    return [
        RetrievedChunk(
            content="员工应在每年一季度完成安全培训。",
            score=0.6,
            source="safety_policy_2026",
            page=2,
            metadata={"source": "safety_policy_2026", "page": 2},
        )
    ]


# 增加于阶段 13.3：运行评测样本并打印 FAR 统计结果。
def run_demo() -> None:
    """使用真实 Pipeline 运行固定样本并计算 False Answer Rate。

    实现方式：为每个问题调用正式 RagPipeline 和本地 grounded fallback，保存实际
    回答为 EvaluationCase，再交给 evaluate_false_answer_rate() 统计。Demo 同时检查
    两个无答案问题均得到固定拒答、可回答问题仍得到 600 元/晚答案，最后打印输入、
    中间统计和最终评测结论；全程不访问外部模型服务。

    参数：
        无入参；使用两个无答案问题和一个有答案问题组成的固定评测集。

    返回：
        无返回值；每条样本结果、False Answer Rate 和验收结论通过标准输出打印。

    异常：
        RuntimeError：样本结果与预期拒答/正常回答不符，或 FAR 评测未达到 0 时抛出。
    """
    pipeline = RagPipeline(
        retriever=_evaluation_retriever,
        llm=GroundedFallbackChatModel(),
    )
    cases: list[EvaluationCase] = []
    print("=== 阶段 13.3 False Answer Rate Demo ===")
    print(f"输入评测集: {len(EVALUATION_INPUTS)} 个问题（2 个无答案，1 个有答案）")
    for index, (question, should_abstain) in enumerate(EVALUATION_INPUTS, start=1):
        result = pipeline.invoke(question)
        cases.append(
            EvaluationCase(
                question=question,
                should_abstain=should_abstain,
                answer=result.answer,
            )
        )
        print(f"样本 {index}: {question}")
        print(f"  预期拒答: {'是' if should_abstain else '否'}")
        print(f"  实际输出: {result.answer}")

    report = evaluate_false_answer_rate(cases)
    answerable_answer = cases[-1].answer
    if report.false_answers or answerable_answer == "根据当前知识库，没有找到足够信息回答该问题。":
        raise RuntimeError("False Answer Rate 验收失败：存在错误回答或正常问题被拒答")

    print(f"评测样本数: {report.total_cases}")
    print(f"无答案样本数: {report.no_answer_cases}")
    print(f"正确拒答数: {report.correct_abstentions}")
    print(f"错误回答数: {report.false_answers}")
    print(f"False Answer Rate: {report.false_answer_rate:.2%}")
    print(f"有答案样本回答: {answerable_answer}")
    print("评测结论: 无答案问题均正确拒答")
    print("False Answer Rate 验收: 通过")


# 增加于阶段 13.3：提供 False Answer Rate Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
