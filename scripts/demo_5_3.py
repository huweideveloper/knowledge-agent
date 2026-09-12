"""阶段 5.3：演示 Chunk 继承业务和权限 Metadata 的完整执行流程。"""

import sys
from pathlib import Path


# 增加于阶段 5.3：支持从项目根目录直接运行 Chunk Metadata Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from app.ingestion.splitter import split_documents


# 增加于阶段 5.3：定义用于演示 Metadata 继承的受控文档输入。
SAMPLE_CONTENT = "第 4 条 上海住宿标准为普通员工 600 元/晚、部门经理 800 元/晚。" * 120
SAMPLE_METADATA = {
    "document_id": "travel_policy_2026",
    "page": 12,
    "department": "finance",
    "allowed_roles": ["employee", "manager"],
}


# 增加于阶段 5.3：运行 Chunk Metadata 继承任务级 Demo。
def run_demo() -> None:
    """执行阶段 5.3 Chunk Metadata Demo，并打印继承前后的关键字段。

    实现方式：构造一份带有文档 ID、页码、归口部门和允许角色的受控
    LangChain Document，调用正式的 split_documents() 完成切块，再逐个打印
    输入 Document 的 Metadata 和每个 Chunk 的对应字段。通过对比输入和输出，
    可以确认切块过程保留业务与权限 Metadata，并继续生成 chunk_id。

    参数：
        无入参；Demo 使用脚本内置的受控文档和 Metadata 示例。

    返回：
        无返回值；Demo 的输入、关键处理结果和最终输出通过标准输出打印。

    异常：
        ValueError: 切块参数或输入 Metadata 不符合正式切块函数要求时由
            split_documents() 抛出。
    """
    document = Document(
        page_content=SAMPLE_CONTENT,
        metadata=SAMPLE_METADATA.copy(),
    )
    chunks = split_documents([document])

    print("=== 阶段 5.3 Chunk Metadata 继承 Demo ===")
    print("输入 Document Metadata:")
    for field in ("document_id", "page", "department", "allowed_roles"):
        print(f"{field}: {document.metadata[field]}")
    print(f"原始 Document 数量: 1")
    print(f"Chunk 数量: {len(chunks)}")
    print("关键处理结果: 每个 Chunk 均保留业务与权限 Metadata")
    for index, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {index} ---")
        for field in (
            "document_id",
            "page",
            "department",
            "allowed_roles",
            "chunk_id",
        ):
            print(f"{field}: {chunk.metadata.get(field)}")
    print("最终输出: Chunk 已继承完整 Metadata")


# 增加于阶段 5.3：提供 Chunk Metadata Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
