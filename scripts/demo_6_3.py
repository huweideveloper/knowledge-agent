"""阶段 6.3：演示 Embedding 维度验收。"""

import sys
from pathlib import Path


# 增加于阶段 6.3：支持从项目根目录直接运行 Embedding 维度 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import (
    embed_documents,
    embed_query,
    get_embedding_model_selection,
    validate_embedding_dimensions,
)


# 增加于阶段 6.3：定义用于演示维度一致性的输入。
SAMPLE_TEXTS = [
    "上海出差住宿上限为普通员工 600 元/晚、部门经理 800 元/晚。",
    "员工无法连接 VPN 时，应先检查网络和多因素认证。",
]
SAMPLE_QUERY = "上海出差住宿标准是多少？"


# 增加于阶段 6.3：运行 Embedding 维度验收任务级 Demo。
def run_demo() -> None:
    """执行阶段 6.3 维度验收 Demo，并打印真实向量的维度结果。

    实现方式：调用正式的 embed_documents() 和 embed_query() 生成文档与 Query
    向量，再调用正式 validate_embedding_dimensions() 检查它们是否处于同一
    向量空间。Demo 打印输入、模型、每组向量长度和最终校验状态，不实现另一套
    Embedding 或维度计算逻辑。

    参数：
        无入参；Demo 使用脚本内置的中文示例文本和 Query。

    返回：
        无返回值；维度验收信息通过标准输出打印。

    异常：
        RuntimeError: Embedding 依赖未安装或模型初始化失败时由正式接口抛出。
        ValueError: 文档向量和 Query 向量维度不一致时由正式校验函数抛出。
    """
    selection = get_embedding_model_selection()
    document_vectors = embed_documents(SAMPLE_TEXTS)
    query_vector = embed_query(SAMPLE_QUERY)
    dimension = validate_embedding_dimensions(document_vectors, query_vector)

    print("=== 阶段 6.3 Embedding 维度验收 Demo ===")
    print(f"输入文本数量: {len(SAMPLE_TEXTS)}")
    print(f"输入 Query: {SAMPLE_QUERY}")
    print(f"模型名称: {selection.model_name}")
    print(f"文档向量数量: {len(document_vectors)}")
    print(f"文档向量维度: {dimension}")
    print(f"Query 向量维度: {len(query_vector)}")
    print("维度一致性: 通过")
    print("关键处理结果: 文档和 Query 向量可以使用同一个向量数据库 Collection")
    print("最终输出: Embedding 维度验收完成")


# 增加于阶段 6.3：提供 Embedding 维度 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
