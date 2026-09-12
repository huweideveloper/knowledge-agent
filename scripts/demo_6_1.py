"""阶段 6.1：演示 Embedding 模型选择。"""

import sys
from pathlib import Path


# 增加于阶段 6.1：支持从项目根目录直接运行 Embedding 模型选择 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import get_embedding_model_selection


# 增加于阶段 6.1：运行 Embedding 模型选择任务级 Demo。
def run_demo() -> None:
    """执行阶段 6.1 模型选择 Demo，并打印可观察的选择依据。

    实现方式：读取正式模型选择配置，打印阶段 5 输出的 Chunk/用户 Query 作为
    输入、模型名称和运行时等关键处理结果，并检查模型类型不是聊天模型。Demo
    不加载模型、不访问网络、不生成向量，避免提前实现阶段 6.2 的职责。

    参数：
        无入参。

    返回：
        无返回值；模型选择信息和最终验收状态通过标准输出打印。

    异常：
        RuntimeError: 配置不是独立 Embedding 模型，或误选聊天模型时抛出。
    """
    selection = get_embedding_model_selection()
    is_dedicated_embedding = selection.model_type == "embedding"
    avoids_chat_model = "deepseek" not in selection.model_name.lower()

    print("=== 阶段 6.1 Embedding 模型选择 Demo ===")
    print("输入: 阶段 5 输出的 Chunk 文本 / 用户 Query")
    print(f"Provider: {selection.provider}")
    print(f"模型名称: {selection.model_name}")
    print(f"运行时: {selection.runtime}")
    print(f"模型类型: {selection.model_type}")
    print(f"适用目的: {selection.purpose}")
    print(f"选择理由: {selection.selection_reason}")
    print(f"不使用聊天模型: {'通过' if is_dedicated_embedding and avoids_chat_model else '失败'}")
    print("关键处理结果: 已确定独立 Embedding 模型，阶段 6.2 再实现实际向量化")

    if not is_dedicated_embedding or not avoids_chat_model:
        raise RuntimeError("Embedding 模型选择验收失败：不能使用聊天模型")

    print("最终输出: Embedding 模型选择完成")


# 增加于阶段 6.1：提供 Embedding 模型选择 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
