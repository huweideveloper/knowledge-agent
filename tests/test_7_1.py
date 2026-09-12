"""阶段 7.1：验证 Docker 启动的 Qdrant 服务可被 Python 客户端访问。"""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.demo_7_1 import start_qdrant_service
from app.vectorstore.qdrant_store import get_qdrant_client


# 增加于阶段 7.1：定义项目根目录，保证测试从任意目录执行时都能定位 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QdrantServiceTest(unittest.TestCase):
    """验证本地 Qdrant 服务启动和 Python 客户端访问链路。"""

    # 增加于阶段 7.4：验证本地客户端跳过已知版本探测警告。
    @patch("app.vectorstore.qdrant_store.QdrantClient")
    def test_local_client_skips_compatibility_probe(self, client_class) -> None:
        """验证本地 Qdrant 客户端关闭不必要的兼容性探测。

        实现方式：替换 QdrantClient 构造器，调用正式连接函数并检查它仍然优先
        使用 gRPC，同时关闭会访问 REST 版本接口的兼容性探测。

        参数：
            client_class: unittest.mock 注入的 QdrantClient 模拟构造器。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：AssertionError：正式连接函数未传递预期客户端配置时抛出。
        """
        get_qdrant_client()

        client_class.assert_called_once_with(
            url="http://localhost:6333",
            prefer_grpc=True,
            check_compatibility=False,
        )

    # 增加于阶段 7.1：验证 Docker 启动超时时返回明确错误。
    @patch(
        "scripts.demo_7_1.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["docker", "compose"], timeout=30),
    )
    def test_docker_start_timeout_raises_runtime_error(self, run_mock) -> None:
        """验证 Docker Compose 长时间无响应时不会让 Demo 无限等待。

        实现方式：让 Docker 子进程模拟达到超时时限，调用正式启动函数并检查
        它将底层 TimeoutExpired 转换为带上下文的 RuntimeError。

        参数：
            run_mock: unittest.mock 注入的 subprocess.run 模拟对象。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：AssertionError：正式函数未转换超时异常或未调用 Docker 子进程时抛出。
        """
        with self.assertRaisesRegex(RuntimeError, "Docker Compose 启动 Qdrant 超时"):
            start_qdrant_service()

        run_mock.assert_called_once()

    # 增加于阶段 7.1：通过真实 Docker 和 qdrant-client 验证服务可访问。
    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is required")
    def test_demo_starts_qdrant_and_python_client_can_access(self) -> None:
        """运行阶段 7.1 Demo，并确认 Python 客户端可以读取集合列表。

        实现方式：从项目根目录启动真实 Demo，由 Demo 调用 Docker Compose 启动
        Qdrant，再通过 qdrant-client 请求 Qdrant 的集合列表接口，最后检查关键
        输出和成功退出码。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 报告错误。

        异常：测试断言失败时抛出 AssertionError。
        """
        script_path = PROJECT_ROOT / "scripts" / "demo_7_1.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("阶段 7.1 本地 Qdrant 服务 Demo", result.stdout)
        self.assertIn("启动方式: Docker Compose", result.stdout)
        self.assertIn("客户端访问: 成功", result.stdout)
        self.assertIn("最终输出: Qdrant 服务可被 Python 客户端访问", result.stdout)


# 增加于阶段 7.1：提供阶段 7.1 自动化测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
