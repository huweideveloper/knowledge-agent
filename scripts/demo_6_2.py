"""阶段 6.2：演示 Embedding 接口封装。"""

import sys
from pathlib import Path


# 增加于阶段 6.2：支持从项目根目录直接运行 Embedding 接口 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import embed_documents, embed_query


# 增加于阶段 6.2：定义用于演示文档和 Query 向量化的输入。
SAMPLE_TEXTS = [
    "上海出差住宿上限为普通员工 600 元/晚、部门经理 800 元/晚。",
    "员工无法连接 VPN 时，应先检查网络和多因素认证。",
]
SAMPLE_QUERY = "上海出差住宿标准是多少？"


# 增加于阶段 6.2：运行 Embedding 接口封装任务级 Demo。
def run_demo() -> None:
    """执行阶段 6.2 Demo，并打印文档和 Query 的实际向量结果。

    实现方式：调用正式的 embed_documents() 和 embed_query() 接口，将两条中文
    文本和一条 Query 交给 6.1 选定的 BAAI/bge-small-zh-v1.5 模型，打印向量数量、长度和
    前几个数值作为可观察输出。Demo 不直接实例化模型，确保它验证的是正式
    封装接口而不是另一套业务逻辑。

    参数：
        无入参；Demo 使用脚本内置的中文示例文本和 Query。

    返回：
        无返回值；输入、向量化结果和最终状态通过标准输出打印。

    异常：
        RuntimeError: Embedding 依赖未安装、模型下载失败或模型初始化失败时
            由正式接口抛出。
        TypeError: Demo 内置输入不符合正式接口约束时由正式接口抛出。
    """
    document_vectors = embed_documents(SAMPLE_TEXTS)
    query_vector = embed_query(SAMPLE_QUERY)

    print("=== 阶段 6.2 Embedding 接口 Demo ===")
    print(f"输入文本数量: {len(SAMPLE_TEXTS)}")
    print(f"输入 Query: {SAMPLE_QUERY}")
    print(f"文档向量数量: {len(document_vectors)}")
    print(f"文档向量长度: {[len(vector) for vector in document_vectors]}")
    print(f"文档向量预览: {[vector[:5] for vector in document_vectors]}")
    print(f"Query 向量已生成，长度: {len(query_vector)}")
    print(f"Query 向量预览: {query_vector[:5]}")
    print("关键处理结果: 文档和 Query 已通过同一 Embedding 封装转换为向量")
    print("最终输出: Embedding 接口封装完成")


# 增加于阶段 6.2：提供 Embedding 接口 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
