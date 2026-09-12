"""阶段 6.1：验证 Embedding 模型选择。"""

import subprocess
import sys
import unittest
from pathlib import Path

from app.embedding.embeddings import get_embedding_model_selection


# 增加于阶段 6.1：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmbeddingModelSelectionTest(unittest.TestCase):
    """验证阶段 6.1 选择独立 Embedding 模型而不是聊天模型。"""

    # 增加于阶段 6.1：验证正式配置选择专用 Embedding 模型。
    def test_selects_dedicated_embedding_model(self) -> None:
        """验证模型名称、运行时和模型职责符合阶段 6.1 要求。

        实现方式：读取正式模型选择配置，检查其使用 Hugging Face 上的
        BAAI/bge-small-zh-v1.5、FastEmbed 运行时和 embedding 模型类型，
        并确认没有把 DeepSeek 聊天模型作为向量模型。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        selection = get_embedding_model_selection()

        self.assertEqual(selection.provider, "huggingface")
        self.assertEqual(selection.model_name, "BAAI/bge-small-zh-v1.5")
        self.assertEqual(selection.runtime, "fastembed")
        self.assertEqual(selection.model_type, "embedding")
        self.assertNotIn("deepseek", selection.model_name.lower())

    # 增加于阶段 6.1：验证任务级 Demo 输出模型选择的关键结果。
    def test_demo_runs_and_reports_model_selection(self) -> None:
        """运行阶段 6.1 Demo，并检查输入、选择结果和最终输出。

        实现方式：从项目根目录启动真实 Demo，检查进程退出码以及模型名称、
        独立 Embedding 类型、运行时和“不使用聊天模型”等可观察信息。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_6_1.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 6.1 Embedding 模型选择 Demo", result.stdout)
        self.assertIn("模型名称: BAAI/bge-small-zh-v1.5", result.stdout)
        self.assertIn("模型类型: embedding", result.stdout)
        self.assertIn("运行时: fastembed", result.stdout)
        self.assertIn("不使用聊天模型: 通过", result.stdout)
        self.assertIn("最终输出: Embedding 模型选择完成", result.stdout)


if __name__ == "__main__":
    unittest.main()
