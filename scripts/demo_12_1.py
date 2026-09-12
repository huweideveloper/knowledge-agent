"""阶段 12.1：演示用户身份信息转换为可判断的 User 对象。"""

import sys
from pathlib import Path


# 增加于阶段 12.1：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.security import User


# 增加于阶段 12.1：执行用户身份输入、Pydantic 建模和结果验收 Demo。
def run_demo() -> None:
    """创建需求文档示例用户并展示校验后的身份对象。

    实现方式：使用固定的用户字典构造 User，导出经过 Pydantic 校验的字段，再与
    预期字典比较并打印输入、关键处理结果和最终验收结论；不连接数据库或检索服务。

    参数：
        无入参；使用需求文档规定的 engineering employee 示例。

    返回：
        无返回值；通过标准输出展示 User 模型处理结果。

    异常：
        RuntimeError：模型导出结果不符合 12.1 验收要求时抛出。
    """
    user_data = {"id": "u001", "department": "engineering", "role": "employee"}
    user = User(**user_data)
    model_output = user.model_dump()
    if model_output != user_data:
        raise RuntimeError("User 模型输出不符合 12.1 验收要求")

    print("=== 阶段 12.1 User 模型 Demo ===")
    print("输入: id=u001, department=engineering, role=employee")
    print("关键处理: 使用 Pydantic 校验身份字段并生成 User 对象")
    print(f"模型输出: {model_output}")
    print("User 模型验收: 通过")


# 增加于阶段 12.1：提供 User 模型 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
