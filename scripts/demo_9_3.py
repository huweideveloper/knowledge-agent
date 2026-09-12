"""阶段 9.3：演示 System Prompt 和 Chat 消息结构。"""

import sys
from pathlib import Path


# 增加于阶段 9.3：支持从项目根目录直接运行 System Prompt Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.prompt import SYSTEM_PROMPT, build_chat_prompt


# 增加于阶段 9.3：定义无需外部模型即可展示 Prompt 的固定输入。
QUESTION = "普通员工去上海出差，酒店最多报多少？"
CONTEXT = "[资料1]\n来源：travel_policy_2026\n正文：普通员工上海住宿上限为 600 元/晚。"


# 增加于阶段 9.3：渲染并打印 System/Human 消息。
def run_demo() -> None:
    """根据固定问题和资料生成 Chat Prompt 并打印关键消息。

    实现方式：调用正式 build_chat_prompt()，读取返回的 LangChain 消息列表，分别
    打印 System Prompt、Human 消息和角色信息，让开发者可以直接检查规则是否进入
    模型请求；本 Demo 不调用外部 LLM。

    参数：
        无入参；使用预设的上海住宿问题和一条示例资料。

    返回：
        无返回值；Prompt 输入、消息结构和验收结论通过标准输出打印。

    异常：
        TypeError、ValueError: 预设 Prompt 输入不符合 build_chat_prompt() 约束时抛出。
    """
    prompt_value = build_chat_prompt(QUESTION, CONTEXT)
    print("=== 阶段 9.3 System Prompt Demo ===")
    print(f"输入问题: {QUESTION}")
    print("关键处理结果: 已生成 LangChain System/Human 消息")
    for message in prompt_value.to_messages():
        print(f"消息角色: {message.type}")
        print(message.content)
        print()
    print("System Prompt 验收: 通过")
    print(f"规则摘要: {SYSTEM_PROMPT.splitlines()[2]}")


# 增加于阶段 9.3：提供 System Prompt Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
