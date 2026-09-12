"""阶段 12.2：演示文档 Metadata 定义 allowed_roles 权限。"""

import sys
from pathlib import Path


# 增加于阶段 12.2：支持从项目根目录或其他工作目录直接运行 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 增加于阶段 12.2：复用正式 Metadata 加载和权限字段校验逻辑。
from app.ingestion.ingest import _load_metadata_records


# 增加于阶段 12.2：定义项目业务 Metadata 目录。
METADATA_DIRECTORY = PROJECT_ROOT / "data" / "metadata"


# 增加于阶段 12.2：执行文档权限读取、校验和结果展示 Demo。
def run_demo() -> None:
    """读取差旅文档的权限 Metadata 并展示允许角色。

    实现方式：调用正式 Metadata 加载函数读取项目全部文档，定位 2026 差旅制度，
    检查其 allowed_roles 与需求示例一致，再打印输入文件、关键处理结果和最终
    验收结论；不执行 Retrieval 或 Qdrant 过滤。

    参数：
        无入参；使用项目 data/metadata/travel_policy_2026.json。

    返回：
        无返回值；通过标准输出展示文档权限定义结果。

    异常：
        RuntimeError：目标文档不存在或允许角色列表不符合 12.2 验收要求时抛出。
    """
    records = _load_metadata_records(METADATA_DIRECTORY)
    travel_policy = next(
        (
            record
            for record in records
            if record.get("document_id") == "travel_policy_2026"
        ),
        None,
    )
    expected_roles = ["employee", "manager", "finance", "admin"]
    if travel_policy is None or travel_policy.get("allowed_roles") != expected_roles:
        raise RuntimeError("差旅文档权限 Metadata 不符合 12.2 验收要求")

    print("=== 阶段 12.2 文档权限 Metadata Demo ===")
    print("输入 Metadata 文件: data/metadata/travel_policy_2026.json")
    print("关键处理: 加载并校验文档 allowed_roles 权限列表")
    print(f"文档: {travel_policy['document_id']}")
    print(f"allowed_roles: {travel_policy['allowed_roles']}")
    print("最终输出: 文档已声明可访问角色")
    print("文档权限验收: 通过")


# 增加于阶段 12.2：提供文档权限 Metadata Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
