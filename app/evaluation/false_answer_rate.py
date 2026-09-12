"""阶段 13.3：统计无答案问题的 False Answer Rate。"""

from collections.abc import Iterable
from dataclasses import dataclass

from app.rag.prompt import ABSTAIN_ANSWER


# 增加于阶段 13.3：定义单条无答案评测样本及其实际系统回答。
@dataclass(frozen=True)
class EvaluationCase:
    """保存一条用于拒答评测的问题、标签和系统回答。"""

    question: str
    should_abstain: bool
    answer: str


# 增加于阶段 13.3：定义 False Answer Rate 评测报告。
@dataclass(frozen=True)
class FalseAnswerRateReport:
    """保存评测总量、拒答正确数和 False Answer Rate。"""

    total_cases: int
    answerable_cases: int
    no_answer_cases: int
    correct_abstentions: int
    false_answers: int
    false_answer_rate: float


# 增加于阶段 13.3：根据固定拒答话术计算无答案问题的错误回答率。
def evaluate_false_answer_rate(
    cases: Iterable[EvaluationCase],
) -> FalseAnswerRateReport:
    """计算评测样本中的 False Answer Rate。

    实现方式：校验每条 EvaluationCase 的问题、拒答标签和实际回答，仅把
    should_abstain 为 True 的样本纳入分母；实际回答不是统一 ABSTAIN_ANSWER 的样本
    计为 false answer，最后返回可直接打印的完整统计报告。该指标专门衡量“本来没
    答案的问题被系统错误回答”的比例，不把有答案问题的正常回答计入 FAR。

    参数：
        cases: 评测样本可迭代对象；每项必须是完整 EvaluationCase，至少包含一条
            should_abstain 为 True 的无答案样本。

    返回：
        FalseAnswerRateReport：包含总样本数、可回答数、无答案数、正确拒答数、错误
            回答数和 0 到 1 之间的 False Answer Rate。

    异常：
        TypeError: cases 不可迭代、元素类型错误、问题或回答不是字符串，或
            should_abstain 不是布尔值时抛出。
        ValueError: cases 为空，问题为空，或没有无答案样本时抛出。
    """
    if isinstance(cases, (str, bytes)):
        raise TypeError("cases must be an iterable of EvaluationCase")
    try:
        case_list = list(cases)
    except TypeError as error:
        raise TypeError("cases must be an iterable of EvaluationCase") from error
    if not case_list:
        raise ValueError("cases must not be empty")
    for case in case_list:
        if not isinstance(case, EvaluationCase):
            raise TypeError("cases must contain only EvaluationCase objects")
        if not isinstance(case.question, str) or not case.question.strip():
            raise ValueError("EvaluationCase.question must not be empty")
        if not isinstance(case.should_abstain, bool):
            raise TypeError("EvaluationCase.should_abstain must be a boolean")
        if not isinstance(case.answer, str):
            raise TypeError("EvaluationCase.answer must be a string")

    no_answer_cases = sum(case.should_abstain for case in case_list)
    if not no_answer_cases:
        raise ValueError("at least one no-answer case is required")
    false_answers = sum(
        case.should_abstain and case.answer.strip() != ABSTAIN_ANSWER
        for case in case_list
    )
    correct_abstentions = no_answer_cases - false_answers
    return FalseAnswerRateReport(
        total_cases=len(case_list),
        answerable_cases=len(case_list) - no_answer_cases,
        no_answer_cases=no_answer_cases,
        correct_abstentions=correct_abstentions,
        false_answers=false_answers,
        false_answer_rate=false_answers / no_answer_cases,
    )
