"""阶段 9.2：验证 Context 最多保留五个检索 Chunk。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.retrieval.vector_retriever import RetrievedChunk

from app.rag.context_builder import limit_chunks


# 增加于阶段 9.2：定义项目根目录，保证 Demo 测试可从任意目录执行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 增加于阶段 9.2：构造有序的候选 Chunk，便于验证截取边界和顺序。
def _make_chunks(count: int) -> list[RetrievedChunk]:
    """创建带连续正文标识的候选 RetrievedChunk 列表。

    实现方式：用正文中的数字表示输入顺序，所有对象共用最小合法 Metadata，
    让测试只观察数量和顺序，不依赖外部向量服务。

    参数：
        count: 需要创建的候选数量，必须为非负整数。

    返回：
        list[RetrievedChunk]：从正文“0”开始按顺序编号的候选结果。

    异常：
        无主动抛出的异常；非法 count 会在 range() 处暴露。
    """
    return [
        RetrievedChunk(
            content=str(index),
            score=1.0 - index / 100,
            source="测试来源",
            page=1,
            metadata={"source": "测试来源", "page": 1},
        )
        for index in range(count)
    ]


class ContextLengthTest(unittest.TestCase):
    """验证候选资料数量被限制在 V1 的五个 Chunk。"""

    # 增加于阶段 9.2：验证超过五个候选时只保留前五个且顺序不变。
    def test_limit_chunks_keeps_only_first_five_without_reordering(self) -> None:
        """确认默认上限为五，且不会对候选结果重新排序。

        实现方式：传入七个按正文编号的候选对象，比较返回数量和正文编号序列。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = limit_chunks(_make_chunks(7))

        self.assertEqual(len(result), 5)
        self.assertEqual([chunk.content for chunk in result], ["0", "1", "2", "3", "4"])

    # 增加于阶段 9.2：验证候选不足上限时不补齐、不丢失原结果。
    def test_limit_chunks_keeps_all_candidates_below_limit(self) -> None:
        """确认少于五个候选时返回全部输入结果。

        实现方式：传入三个对象并比较完整列表，确保函数只负责上限控制，不生成
        人工占位对象，也不改变已有对象内容。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        chunks = _make_chunks(3)
        self.assertEqual(limit_chunks(chunks), chunks)

    # 增加于阶段 9.2：验证数量上限的输入约束。
    def test_limit_chunks_rejects_non_positive_or_non_integer_limit(self) -> None:
        """确认零、负数、布尔值和浮点数不会被当作合法上限。

        实现方式：分别调用 limit_chunks()，检查每个非法值都抛出与接口约定一致的
        异常，防止错误配置悄悄产生空 Context 或隐式类型转换。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        for invalid_limit in (0, -1):
            with self.assertRaises(ValueError):
                limit_chunks(_make_chunks(1), invalid_limit)
        for invalid_limit in (True, 1.5):
            with self.assertRaises(TypeError):
                limit_chunks(_make_chunks(1), invalid_limit)

    # 增加于阶段 9.2：验证阶段 Demo 展示候选数量到受控数量的变化。
    def test_demo_limits_real_candidates_to_five(self) -> None:
        """运行阶段 9.2 Demo，并检查七个候选最终只保留五个。

        实现方式：启动真实 Demo，由其从 Qdrant 召回七条结果并调用正式 limit_chunks；
        断言输出包含输入、限制值、输出数量和验收标记。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_9_2.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 9.2 Context 长度控制 Demo", result.stdout)
        self.assertIn("输入候选数量: 7", result.stdout)
        self.assertIn("最大保留数量: 5", result.stdout)
        self.assertIn("受控 Chunk 数量: 5", result.stdout)
        self.assertIn("Context 长度控制验收: 通过", result.stdout)


# 增加于阶段 9.2：提供 Context 长度控制测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
