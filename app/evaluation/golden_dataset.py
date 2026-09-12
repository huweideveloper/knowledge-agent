"""阶段 14.1：加载并校验可重复使用的 Golden Dataset。"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


# 增加于阶段 14.1：定义项目内默认 Golden Dataset 文件位置。
DEFAULT_GOLDEN_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_dataset.json"


# 增加于阶段 14.1：定义一条包含答案和来源 ground truth 的标准题库记录。
@dataclass(frozen=True)
class GoldenRecord:
    """保存后续 Recall、MRR 和答案评测所需的一条标准 QA 记录。"""

    question: str
    expected_answer: str
    expected_document: str
    expected_page: int


# 增加于阶段 14.1：从 JSON 文件读取并严格校验 Golden Dataset。
def load_golden_dataset(
    path: str | Path = DEFAULT_GOLDEN_DATASET_PATH,
) -> tuple[GoldenRecord, ...]:
    """加载项目 Golden Dataset 并转换为不可变标准记录。

    实现方式：校验路径类型和文件存在性，使用标准库按 UTF-8 解析 JSON，检查顶层
    列表、记录字段、非空文本、正整数页码和问题唯一性，最后按文件顺序构造
    GoldenRecord 元组；函数不修改数据文件，返回结果可直接供后续指标使用。

    参数：
        path: Golden Dataset JSON 文件路径，可传字符串或 Path；默认读取项目
            data/golden_dataset.json。

    返回：
        tuple[GoldenRecord, ...]：按 JSON 顺序排列的只读题库记录；空题库不会返回。

    异常：
        TypeError: path 不是字符串或 Path，JSON 记录不是对象，或字段类型不正确时抛出。
        ValueError: 文件不是合法 JSON、顶层不是非空列表、字段缺失/为空、页码非正数、
            存在额外字段或问题重复时抛出。
        FileNotFoundError: path 指向的文件不存在时抛出。
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path")
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Golden Dataset 不是合法 JSON：{dataset_path}") from error

    if not isinstance(payload, list) or not payload:
        raise ValueError("Golden Dataset 顶层必须是非空数组")
    required_fields = {
        "question",
        "expected_answer",
        "expected_document",
        "expected_page",
    }
    records: list[GoldenRecord] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"Golden Dataset 第 {index} 条记录必须是对象")
        if set(item) != required_fields:
            raise ValueError(
                f"Golden Dataset 第 {index} 条记录字段必须为"
                " question/expected_answer/expected_document/expected_page"
            )

        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_document = item["expected_document"]
        expected_page = item["expected_page"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Golden Dataset 第 {index} 条 question 不能为空")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(f"Golden Dataset 第 {index} 条 expected_answer 不能为空")
        if not isinstance(expected_document, str) or not expected_document.strip():
            raise ValueError(f"Golden Dataset 第 {index} 条 expected_document 不能为空")
        if (
            isinstance(expected_page, bool)
            or not isinstance(expected_page, int)
            or expected_page <= 0
        ):
            raise ValueError(f"Golden Dataset 第 {index} 条 expected_page 必须是正整数")

        normalized_question = question.strip()
        if normalized_question in seen_questions:
            raise ValueError(f"Golden Dataset 存在重复 question：{normalized_question}")
        seen_questions.add(normalized_question)
        records.append(
            GoldenRecord(
                question=normalized_question,
                expected_answer=expected_answer.strip(),
                expected_document=expected_document.strip(),
                expected_page=expected_page,
            )
        )
    return tuple(records)
