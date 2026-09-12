"""阶段 11.2：演示按 session_id 保存和读取 Conversation History。"""

import sys
from pathlib import Path


# 增加于阶段 11.2：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation import ConversationHistoryStore, create_session_id


# 增加于阶段 11.2：执行会话历史写入、读取和验收 Demo。
def run_demo() -> None:
    """保存一轮 user/assistant 消息并打印实际读取结果。

    实现方式：创建一个新 session_id，使用 SQLAlchemy 内存数据库保存两条消息，按
    同一会话读取后检查角色顺序和正文，再将输入与读取结果打印出来；Demo 不依赖
    外部 PostgreSQL 服务，但存储接口接受 PostgreSQL URL。

    参数：
        无入参；使用固定的上海住宿问题和示例回答。

    返回：
        无返回值；通过标准输出展示 Conversation History 的保存结果。

    异常：
        RuntimeError: 读取结果与写入的 user/assistant 消息不一致时抛出。
    """
    session_id = create_session_id()
    user_content = "上海住宿标准是多少？"
    assistant_content = "普通员工为 600 元/晚。"
    store = ConversationHistoryStore()
    try:
        store.append_message(session_id, "user", user_content)
        store.append_message(session_id, "assistant", assistant_content)
        history = store.get_messages(session_id)
    finally:
        store.close()

    if [message.role for message in history] != ["user", "assistant"]:
        raise RuntimeError("Conversation History 角色顺序错误")
    if [message.content for message in history] != [user_content, assistant_content]:
        raise RuntimeError("Conversation History 正文读取错误")

    print("=== 阶段 11.2 Conversation History Demo ===")
    print(f"输入 session_id: {session_id}")
    print(f"输入 user: {user_content}")
    print(f"输入 assistant: {assistant_content}")
    print("关键处理: 使用 SQLAlchemy 按 session_id 保存并读取消息")
    print(f"读取消息数量: {len(history)}")
    print(f"读取角色顺序: {' -> '.join(message.role for message in history)}")
    print("Conversation History 保存验收: 通过")


# 增加于阶段 11.2：提供会话历史 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
