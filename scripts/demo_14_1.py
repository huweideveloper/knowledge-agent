"""阶段 14.1：演示 Golden Dataset 的加载和字段校验。"""

import sys
from pathlib import Path


# 增加于阶段 14.1：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.golden_dataset import (
    DEFAULT_GOLDEN_DATASET_PATH,
    load_golden_dataset,
)


# 增加于阶段 14.1：运行 Golden Dataset 加载、字段校验和代表性样本展示。
def run_demo() -> None:
    """加载项目 Golden Dataset 并打印可直接观察的 ground truth。

    实现方式：调用正式 load_golden_dataset() 读取 JSON，检查题库数量、样本字段和
    首条记录的答案与来源信息，再打印文件、处理结果和后续指标可复用的输出；Demo
    不访问 Qdrant、Embedding 或外部模型服务。

    参数：
        无入参；使用项目默认 data/golden_dataset.json。

    返回：
        无返回值；题库输入、关键校验结果和验收结论通过标准输出打印。

    异常：
        RuntimeError：题库数量或首条 ground truth 不符合本阶段预期时抛出。
        FileNotFoundError、TypeError、ValueError：默认题库文件读取或校验失败时透传。
    """
    records = load_golden_dataset()
    first = records[0]
    if len(records) != 3:
        raise RuntimeError("Golden Dataset 验收失败：题目数量不是 3")
    if (
        first.question != "普通员工去上海出差，酒店最多报多少？"
        or first.expected_document != "travel_policy_2026"
        or first.expected_page != 1
    ):
        raise RuntimeError("Golden Dataset 验收失败：首条 ground truth 不符合预期")

    relative_path = DEFAULT_GOLDEN_DATASET_PATH.relative_to(PROJECT_ROOT)
    print("=== 阶段 14.1 Golden Dataset Demo ===")
    print(f"输入数据文件: {relative_path}")
    print(f"题目数量: {len(records)}")
    print("字段校验: question / expected_answer / expected_document / expected_page")
    print("关键处理结果: JSON 已解析并逐条校验，问题均唯一且页码为正整数")
    print(f"示例问题: {first.question}")
    print(f"示例期望答案: {first.expected_answer}")
    print(f"示例期望文档: {first.expected_document}")
    print(f"示例期望页码: {first.expected_page}")
    print("最终输出: Golden Dataset 已准备好供后续 Recall、MRR 和答案指标复用")
    print("Golden Dataset 验收: 通过")


# 增加于阶段 14.1：提供 Golden Dataset Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
