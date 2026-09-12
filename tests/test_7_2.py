"""阶段 7.2：验证 Qdrant Collection 创建及 Embedding 维度配置。"""

import subprocess
import sys
import unittest
from pathlib import Path


# 增加于阶段 7.2：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QdrantCollectionTest(unittest.TestCase):
    """验证 enterprise_knowledge Collection 的创建和配置。"""

    # 增加于阶段 7.2：通过真实 Qdrant 和 Embedding 验证 Collection 配置。
    def test_demo_creates_collection_with_embedding_dimension(self) -> None:
        """运行阶段 7.2 Demo，并确认 Collection 维度与 Embedding 一致。

        实现方式：从项目根目录启动真实 Demo，由 Demo 生成当前 Embedding 维度、
        创建或校验 enterprise_knowledge Collection，再检查关键配置和成功退出码。
        重复运行使用同一 Collection，验证实现不会重复创建或删除数据。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_2.py"
        first_result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        second_result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        for result in (first_result, second_result):
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("阶段 7.2 Collection 创建 Demo", result.stdout)
            self.assertIn("Collection 名称: enterprise_knowledge", result.stdout)
            self.assertIn("Embedding 向量维度: 512", result.stdout)
            self.assertIn("Collection 向量维度: 512", result.stdout)
            self.assertIn("距离度量: Cosine", result.stdout)
            self.assertIn("维度一致性: 通过", result.stdout)
            self.assertIn("最终输出: Collection 创建并配置完成", result.stdout)


# 增加于阶段 7.2：提供阶段 7.2 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
