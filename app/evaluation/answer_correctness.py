"""阶段 14.4：计算生成答案与期望答案的 Answer Correctness。"""

import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.evaluation.golden_dataset import GoldenRecord


# 增加于阶段 14.4：定义答案正确性评测报告。
@dataclass(frozen=True)
class AnswerCorrectnessReport:
    """保存逐题答案正确性分数、正确题数和平均分。"""

    total_questions: int
    scores: tuple[float, ...]
    correct_questions: int
    correctness_score: float


# 增加于阶段 14.4：统一答案格式，减少空白、标点和 Unicode 形式差异的干扰。
def normalize_answer(answer: str) -> str:
    """将答案转换为适合 V1 精确比较的标准形式。

    实现方式：先进行 Unicode NFKC 归一化和大小写折叠，再移除空白字符与 Unicode
    标点；该处理让“600 元/晚”和“600元/晚”等格式差异不影响比较，但不进行同义词、
    语义改写或事实推理判断。

    参数：
        answer: 待标准化的生成答案或期望答案，必须是字符串。

    返回：
        str：去除格式差异后的比较文本；输入为空白字符串时返回空字符串。

    异常：
        TypeError：answer 不是字符串时抛出。
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


# 增加于阶段 14.4：根据 Golden Dataset 比较生成答案并计算平均正确性。
# ponytail: V1 仅做标准化后的精确匹配；同义改写无法识别，后续可升级为模型评审。
def evaluate_answer_correctness(
    golden_records: Iterable[GoldenRecord],
    generated_answers: Mapping[str, str],
) -> AnswerCorrectnessReport:
    """计算生成答案相对 Golden Dataset 期望答案的正确性分数。

    实现方式：校验 GoldenRecord 序列和按问题索引的生成答案，逐题对期望答案与生成
    答案执行 normalize_answer() 后的精确比较；匹配记为 1.0，不匹配或缺少生成答案
    记为 0.0，最后返回逐题分数、正确题数和平均 Correctness Score。该 V1 评测不
    调用外部 LLM，也不把来源文档支持关系混入本任务，Faithfulness 留给 14.5。

    参数：
        golden_records: 14.1 Golden Dataset 记录，可迭代且不能为空，问题必须唯一。
        generated_answers: 问题到最终生成答案的映射；缺少某题答案按错误答案计分。

    返回：
        AnswerCorrectnessReport：包含总题数、按题库顺序排列的 0.0/1.0 分数、正确
            题数和 0 到 1 之间的平均 correctness_score。

    异常：
        TypeError：题库、生成答案映射或生成答案类型不符合接口约束时抛出。
        ValueError：题库为空、问题重复或期望答案为空时抛出。
    """
    if isinstance(golden_records, (str, bytes)):
        raise TypeError("golden_records must be an iterable of GoldenRecord")
    if not isinstance(generated_answers, Mapping):
        raise TypeError("generated_answers must be a mapping")
    try:
        records = list(golden_records)
    except TypeError as error:
        raise TypeError("golden_records must be an iterable of GoldenRecord") from error
    if not records:
        raise ValueError("golden_records must not be empty")

    questions: set[str] = set()
    for record in records:
        if not isinstance(record, GoldenRecord):
            raise TypeError("golden_records must contain only GoldenRecord objects")
        if not isinstance(record.question, str) or not record.question.strip():
            raise ValueError("GoldenRecord.question must not be empty")
        if not isinstance(record.expected_answer, str) or not record.expected_answer.strip():
            raise ValueError("GoldenRecord.expected_answer must not be empty")
        if record.question in questions:
            raise ValueError(f"golden_records contains duplicate question: {record.question}")
        questions.add(record.question)

    scores: list[float] = []
    for record in records:
        generated_answer = generated_answers.get(record.question, "")
        if not isinstance(generated_answer, str):
            raise TypeError("generated_answers values must be strings")
        score = float(
            normalize_answer(generated_answer)
            == normalize_answer(record.expected_answer)
        )
        scores.append(score)

    total_questions = len(records)
    correct_questions = sum(score == 1.0 for score in scores)
    return AnswerCorrectnessReport(
        total_questions=total_questions,
        scores=tuple(scores),
        correct_questions=correct_questions,
        correctness_score=sum(scores) / total_questions,
    )
