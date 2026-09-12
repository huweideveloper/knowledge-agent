"""阶段 9.4：验证 DeepSeek 配置统一从项目 .env 文件读取。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.config import get_deepseek_api_key


class DeepSeekConfigTest(unittest.TestCase):
    """验证 .env 中 DeepSeek Key 的读取和清洗规则。"""

    # 增加于阶段 9.4：验证配置读取指定 .env 文件中的非空 Key。
    def test_reads_deepseek_key_from_env_file(self) -> None:
        """确认配置函数读取值并去除包裹引号与首尾空白。

        实现方式：创建临时 .env 文件，写入带双引号的配置，再调用正式配置入口；
        测试不触碰项目真实密钥，也不访问网络。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OTHER=value\nDEEPSEEK_API_KEY=\"  test-key  \"\n",
                encoding="utf-8",
            )

            self.assertEqual(get_deepseek_api_key(env_path), "  test-key  ")

    # 增加于阶段 9.4：验证空值和缺少文件不产生虚假 Key。
    def test_returns_none_for_missing_or_empty_key(self) -> None:
        """确认缺少 .env 或 Key 为空时返回 None。

        实现方式：先读取不存在的路径，再读取包含空配置项的临时文件，确保默认
        模型会选择本地 fallback，而不是尝试使用空密钥请求远程 API。

        参数：无入参。

        返回：无返回值；断言失败时由 unittest 抛出 AssertionError。

        异常：测试断言失败时抛出 AssertionError。
        """
        with TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            self.assertIsNone(get_deepseek_api_key(env_path))
            env_path.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")
            self.assertIsNone(get_deepseek_api_key(env_path))


# 增加于阶段 9.4：提供配置读取测试的命令行入口。
if __name__ == "__main__":
    unittest.main()
