"""阶段 9.4：项目本地 .env 配置读取。"""

from pathlib import Path


# 增加于阶段 9.4：定义项目根目录下的默认 .env 路径。
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


# 增加于阶段 9.4：去除 .env 值外层引号但保留值内部空格。
def _strip_outer_quotes(value: str) -> str:
    """去除配置值成对的单引号或双引号。

    实现方式：检查值首尾是否由同一种引号包裹；只有成对包裹时去除外层，避免
    把普通字符串中的引号或内部空格错误改写。

    参数：
        value: 从 .env 等号右侧读取的原始字符串。

    返回：
        str：去除成对外层引号后的配置值。

    异常：
        TypeError: value 不是字符串时抛出。
    """
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


# 增加于阶段 9.4：从项目 .env 读取 DeepSeek API Key。
def get_deepseek_api_key(env_path: Path | None = None) -> str | None:
    """从项目根目录 .env 读取非空 DEEPSEEK_API_KEY。

    实现方式：读取指定或默认项目 `.env`，逐行跳过空行和注释，解析
    `DEEPSEEK_API_KEY=...` 配置并去除值两端空白和外层引号；本函数不读取 shell
    环境变量、不打印密钥，也不修改进程环境。env_path 参数仅用于测试或显式配置。

    参数：
        env_path: 可选 .env 文件路径；未提供时使用项目根目录 `.env`。

    返回：
        str | None：非空 API Key；文件不存在、没有该配置或配置为空时返回 None。

    异常：
        TypeError: env_path 不是 Path 或字符串时抛出。
        RuntimeError: .env 存在但无法读取时抛出。
    """
    if env_path is None:
        path = DEFAULT_ENV_PATH
    elif isinstance(env_path, (str, Path)):
        path = Path(env_path)
    else:
        raise TypeError("env_path must be a path or None")

    if not path.exists():
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"无法读取配置文件：{path}") from error

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "DEEPSEEK_API_KEY":
            continue
        cleaned_value = _strip_outer_quotes(value.strip())
        return cleaned_value if cleaned_value else None
    return None
