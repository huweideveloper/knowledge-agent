"""阶段 14.4：演示生成答案的 Answer Correctness 计算。"""

import sys
from pathlib import Path


# 增加于阶段 14.4：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.answer_correctness import evaluate_answer_correctness
from app.evaluation.golden_dataset import load_golden_dataset


# 增加于阶段 14.4：定义不访问外部 LLM 的固定生成答案，覆盖正确、错误和格式差异。
GENERATED_ANSWERS = (
    "普通员工去上海出差的酒店最多可报销600元/晚",
    "出差前需要提交出差申请。",
    "员工应在每年一季度完成安全培训。",
)


# 增加于阶段 14.4：运行答案正确性评测并打印输入、逐题分数和汇总结果。
def run_demo() -> None:
    """读取 Golden Dataset 并演示生成答案是否符合期望答案。

    实现方式：加载 14.1 题库，将固定生成答案按题目顺序映射为评测输入，调用正式
    evaluate_answer_correctness()；第一题验证空格和标点差异仍可匹配，第二题验证
    缺少审批要求会被判错，第三题验证完整答案命中，最后打印输入、关键处理结果和
    验收结论。Demo 不调用外部 LLM，便于稳定复现答案正确性评测。

    参数：
        无入参；使用项目默认 Golden Dataset 和三条固定生成答案。

    返回：
        无返回值；期望答案、生成答案、逐题分数、平均 Correctness Score 和验收结论
        通过标准输出打印。

    异常：
        RuntimeError：题库数量、生成答案数量、逐题分数或平均正确性不符合预期时抛出。
    """
    records = load_golden_dataset()
    if len(records) != len(GENERATED_ANSWERS):
        raise RuntimeError("Answer Correctness 验收失败：题库与生成答案数量不一致")
    generated_answers = dict(zip((record.question for record in records), GENERATED_ANSWERS))
    report = evaluate_answer_correctness(records, generated_answers)
    expected_scores = (1.0, 0.0, 1.0)
    if (
        report.total_questions != 3
        or report.scores != expected_scores
        or report.correct_questions != 2
        or report.correctness_score != 2 / 3
    ):
        raise RuntimeError("Answer Correctness 验收失败：答案匹配或指标统计不符合预期")

    print("=== 阶段 14.4 Answer Correctness Demo ===")
    print(f"Golden Dataset 题目数: {report.total_questions}")
    print("输入字段: LLM 生成答案 + Golden Dataset 期望答案")
    print("评分方式: Unicode 标准化后忽略空白和标点进行精确匹配")
    for index, (record, score) in enumerate(zip(records, report.scores), start=1):
        print(f"样本 {index} 问题: {record.question}")
        print(f"  期望答案: {record.expected_answer}")
        print(f"  LLM 生成答案: {generated_answers[record.question]}")
        print(f"样本 {index} Correctness Score: {score:.2f}")
        print(f"样本 {index} 判定: {'正确' if score == 1.0 else '错误'}")
    print(f"正确答案数: {report.correct_questions}")
    print(f"Answer Correctness: {report.correctness_score:.2%}")
    print("关键处理结果: 格式差异不影响匹配，缺少审批要求的答案被识别为错误")
    print("Answer Correctness 验收: 通过")


# 增加于阶段 14.4：提供 Answer Correctness Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
