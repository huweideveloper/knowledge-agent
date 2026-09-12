"""阶段 7.1：演示 Docker 启动 Qdrant 并由 Python 客户端访问。"""

import subprocess
import sys
from pathlib import Path


# 增加于阶段 7.1：支持从项目根目录直接运行 Qdrant 服务 Demo。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vectorstore.qdrant_store import (
    DEFAULT_QDRANT_URL,
    get_qdrant_client,
    wait_for_qdrant,
)


# 增加于阶段 7.1：通过 Docker Compose 启动本地 Qdrant 服务。
# 修改于阶段 7.1：增加 Docker 启动超时保护，避免镜像下载异常时无限等待。
def start_qdrant_service() -> None:
    """启动项目定义的 Qdrant Docker Compose 服务。

    实现方式：执行 docker compose up -d qdrant，复用已有容器或创建新容器，
    不删除数据卷；命令失败时将 Docker 错误转换为带上下文的 RuntimeError。

    参数：
        无入参；Compose 文件固定为项目根目录下的 docker-compose.yml。

    返回：
        无返回值；命令成功表示容器已进入后台启动流程。

    异常：
        RuntimeError: Docker Compose 不可用、镜像拉取失败、服务启动命令失败或
            超过 30 秒未返回。
    """
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "up",
                "-d",
                "qdrant",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Docker Compose 启动 Qdrant 超时") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Docker Compose 启动 Qdrant 失败：{detail}")


# 增加于阶段 7.1：运行 Qdrant 服务可访问性的任务级验收 Demo。
def run_demo() -> None:
    """启动本地 Qdrant 并验证 Python 客户端可以读取集合列表。

    实现方式：先通过项目 Docker Compose 文件启动 Qdrant，再创建正式的
    QdrantClient 并轮询只读集合列表接口，打印服务地址、Collection 列表和
    客户端访问结果。本 Demo 不创建 Collection，创建 Collection 属于阶段 7.2。

    参数：
        无入参；服务配置来自项目根目录的 docker-compose.yml。

    返回：
        无返回值；启动和访问结果通过标准输出打印。

    异常：
        RuntimeError: Docker Compose 启动失败，或 Qdrant 在限定时间内不可访问。
        TypeError、ValueError: Qdrant 客户端配置不符合正式接口约束时抛出。
    """
    start_qdrant_service()
    client = get_qdrant_client()
    collection_names = wait_for_qdrant(client)

    print("=== 阶段 7.1 本地 Qdrant 服务 Demo ===")
    print("输入: Docker Compose Qdrant 服务配置")
    print(f"服务地址: {DEFAULT_QDRANT_URL}")
    print("启动方式: Docker Compose")
    print(f"Collection 列表: {collection_names}")
    print("关键处理结果: Python 客户端已成功请求 Qdrant 集合列表 API")
    print("客户端访问: 成功")
    print("最终输出: Qdrant 服务可被 Python 客户端访问")


# 增加于阶段 7.1：提供 Qdrant 服务 Demo 的命令行入口。
if __name__ == "__main__":
    run_demo()
