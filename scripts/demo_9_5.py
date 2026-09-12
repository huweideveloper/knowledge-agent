"""阶段 9.5：运行第一次端到端 RAG 问答并验证答案来源。"""

import sys
from pathlib import Path


# 增加于阶段 9.5：支持从项目根目录直接运行第一次端到端 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.llm import get_default_chat_model
from app.rag.pipeline import RagPipeline


# 增加于阶段 9.5：定义需求文档规定的第一次端到端问题和验收证据。
QUESTION = "普通员工去上海出差，酒店最多报多少？"
EXPECTED_EVIDENCE = "600 元/晚"


# 增加于阶段 9.5：执行完整 RAG 并检查回答确实使用检索资料。
def run_demo() -> None:
    """检索上海住宿政策并生成基于资料的端到端回答。

    实现方式：调用默认 DeepSeek 或无密钥时的 grounded fallback，运行
    RagPipeline，检查 Context 包含 600 元/晚且回答同时包含金额和“元/晚”单位，
    最后打印问题、检索 Context、模型回答和来源验收结果。

    参数：
        无入参；使用需求文档规定的问题“普通员工去上海出差，酒店最多报多少？”。

    返回：
        无返回值；端到端处理结果和验收结论通过标准输出打印。

    异常：
        RuntimeError: 未召回资料、Context 缺少预期证据或模型回答缺少该证据时抛出。
        TypeError、ValueError: Pipeline 输入不符合约束时抛出。
    """
    model = get_default_chat_model()
    result = RagPipeline(llm=model).invoke(QUESTION)
    evidence_in_context = EXPECTED_EVIDENCE in result.context
    evidence_in_answer = "600" in result.answer and "元/晚" in result.answer
    if not result.chunks:
        raise RuntimeError("端到端验收失败：没有召回资料")
    if not evidence_in_context:
        raise RuntimeError("端到端验收失败：检索 Context 缺少 600 元/晚")
    if not evidence_in_answer:
        raise RuntimeError("端到端验收失败：模型回答缺少检索证据 600 元/晚")

    print("=== 阶段 9.5 第一次端到端测试 Demo ===")
    print(f"输入问题: {QUESTION}")
    print(f"模型适配器: {type(model).__name__}")
    print(f"检索 Chunk 数量: {len(result.chunks)}")
    print("检索 Context:")
    print(result.context)
    print("最终回答:")
    print(result.answer)
    print(f"答案来自检索 Context: {'是' if evidence_in_context and evidence_in_answer else '否'}")
    print("端到端验收: 通过")


# 增加于阶段 9.5：提供第一次端到端 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
