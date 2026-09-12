"""阶段 11.4：演示基于 Conversation History 补全指代问题。"""

import sys
from pathlib import Path


# 增加于阶段 11.4：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation import (
    ConversationHistoryStore,
    create_session_id,
    rewrite_follow_up_question,
)


# 增加于阶段 11.4：执行历史问题、指代问题和改写结果验收 Demo。
def run_demo() -> None:
    """保存需求示例对话并展示“那经理呢？”的完整改写结果。

    实现方式：创建会话并保存一轮 user/assistant 历史，从存储中读取历史，再调用
    正式指代改写函数生成完整问题；最后检查并打印历史、当前问题和改写结果。

    参数：
        无入参；使用需求文档规定的上海住宿标准示例。

    返回：
        无返回值；通过标准输出展示指代问题处理结果。

    异常：
        RuntimeError: 改写结果不是需求文档规定的完整问题时抛出。
    """
    session_id = create_session_id()
    history_store = ConversationHistoryStore()
    try:
        history_store.append_message(session_id, "user", "上海普通员工住宿标准多少？")
        history_store.append_message(session_id, "assistant", "600元。")
        history = history_store.get_messages(session_id)
    finally:
        history_store.close()

    current_question = "那经理呢？"
    rewritten_question = rewrite_follow_up_question(current_question, history)
    expected_question = "经理在上海出差住宿标准是多少？"
    if rewritten_question != expected_question:
        raise RuntimeError("指代问题改写结果不符合要求")

    print("=== 阶段 11.4 指代问题 Demo ===")
    print(f"输入 session_id: {session_id}")
    print(f"历史问题: {history[0].content}")
    print(f"历史回答: {history[1].content}")
    print(f"当前问题: {current_question}")
    print(f"关键处理: 从历史 user 问题提取城市并补全省略语义")
    print(f"改写问题: {rewritten_question}")
    print("指代问题验收: 通过")


# 增加于阶段 11.4：提供指代问题 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
