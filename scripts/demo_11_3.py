"""阶段 11.3：演示 Conversation History 暂不限制历史长度。"""

import sys
from pathlib import Path


# 增加于阶段 11.3：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation import ConversationHistoryStore, create_session_id


# 增加于阶段 11.3：执行多轮历史写入、完整读取和结果验收 Demo。
def run_demo() -> None:
    """写入 12 轮消息并展示未限制的完整历史。

    实现方式：创建新会话并写入连续编号的 user/assistant 消息，读取全部历史，检查
    首尾内容后打印写入和读取轮数；Demo 使用 SQLite 内存库，不依赖外部数据库服务。

    参数：
        无入参；使用 12 轮编号消息作为可观察的演示输入。

    返回：
        无返回值；通过标准输出展示完整历史读取结果。

    异常：
        RuntimeError: 读取结果不是完整 12 轮或返回顺序错误时抛出。
    """
    session_id = create_session_id()
    store = ConversationHistoryStore()
    try:
        for index in range(12):
            store.append_message(session_id, "user", f"user-{index}")
            store.append_message(session_id, "assistant", f"assistant-{index}")
        history = store.get_recent_messages(session_id)
    finally:
        store.close()

    if len(history) != 24:
        raise RuntimeError("完整历史读取缺少消息")
    if history[0].content != "user-0" or history[-1].content != "assistant-11":
        raise RuntimeError("完整历史读取顺序错误")

    print("=== 阶段 11.3 Conversation History 完整历史 Demo ===")
    print(f"输入 session_id: {session_id}")
    print("输入历史轮数: 12")
    print("关键处理: 暂不限制上下文长度，读取全部会话历史")
    print("读取历史轮数: 12")
    print(f"读取消息数量: {len(history)}")
    print(f"最早读取消息: {history[0].content}")
    print(f"最新读取消息: {history[-1].content}")
    print("完整历史读取验收: 通过")


# 增加于阶段 11.3：提供完整历史读取 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
