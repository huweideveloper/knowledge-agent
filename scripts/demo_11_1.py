"""阶段 11.1：演示新会话生成独立 session_id。"""

import re
import sys
from pathlib import Path


# 增加于阶段 11.1：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.conversation import create_session_id


# 增加于阶段 11.1：执行新会话 session_id 生成和格式验收 Demo。
def run_demo() -> None:
    """创建新会话并打印输入、关键处理结果和最终 session_id。

    实现方式：调用正式会话标识生成函数，用需求规定的正则格式检查返回值，随后
    输出会话身份和验收结论；本 Demo 不保存历史，也不连接数据库。

    参数：
        无入参；使用固定的“新会话”演示输入。

    返回：
        无返回值；通过标准输出展示 11.1 的处理结果。

    异常：
        RuntimeError: 生成的 session_id 不符合 `session_<UUID>` 格式时抛出。
    """
    session_id = create_session_id()
    if not re.fullmatch(r"session_[0-9a-f]{32}", session_id):
        raise RuntimeError("session_id 格式不符合要求")

    print("=== 阶段 11.1 会话 session_id Demo ===")
    print("输入: 新会话")
    print("关键处理: 使用 UUID 生成独立会话身份")
    print(f"输出 session_id: {session_id}")
    print("session_id 生成验收: 通过")


# 增加于阶段 11.1：提供会话标识 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
