"""阶段 6.4：演示 Embedding 的语义相似度实验。"""

import sys
from pathlib import Path


# 增加于阶段 6.4：支持从项目根目录直接运行语义相似度 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embeddings import cosine_similarity, embed_documents


# 增加于阶段 6.4：定义文档规定的相似度实验输入。
EXPERIMENT_TEXTS = {
    "A": "上海出差住宿上限",
    "B": "上海酒店报销标准",
    "C": "VPN密码如何修改",
}


# 增加于阶段 6.4：运行真实 Embedding 的语义相似度验收 Demo。
def run_demo() -> None:
    """生成 A/B/C 向量并验证相近语义的相似度更高。

    实现方式：按文档顺序将 A、B、C 文本交给正式 Embedding 接口，再用正式的
    余弦相似度函数计算 A/B 和 A/C，最后断言 A/B 得分更高并打印实验结果。
    Demo 不使用手工分数，也不引入向量数据库。

    参数：
        无入参；实验使用文档规定的三条内置中文文本。

    返回：
        无返回值；实验输入、相似度和验收结果通过标准输出打印。

    异常：
        RuntimeError: A/B 相似度没有高于 A/C 时抛出，表示模型未体现预期的
            基本语义相似性。
        ValueError: 向量维度不一致或出现零向量时由相似度函数抛出。
        TypeError: Embedding 输入或向量元素类型不符合要求时由正式接口抛出。
    """
    labels = list(EXPERIMENT_TEXTS)
    vectors = embed_documents([EXPERIMENT_TEXTS[label] for label in labels])
    similarity_ab = cosine_similarity(vectors[0], vectors[1])
    similarity_ac = cosine_similarity(vectors[0], vectors[2])

    print("=== 阶段 6.4 语义相似度实验 Demo ===")
    for label in labels:
        print(f"{label}：{EXPERIMENT_TEXTS[label]}")
    print(f"Similarity(A,B): {similarity_ab:.6f}")
    print(f"Similarity(A,C): {similarity_ac:.6f}")

    if similarity_ab <= similarity_ac:
        raise RuntimeError("语义相似性验收失败：Similarity(A,B) 未高于 Similarity(A,C)")

    print("关键处理结果: A 与 B 的相似度高于 A 与 C")
    print("语义相似性验收: 通过")
    print("最终输出: 语义相似度实验完成")


# 增加于阶段 6.4：提供语义相似度 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
