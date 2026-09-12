"""阶段 9.4：DeepSeek 聊天模型适配和无密钥的本地 grounded fallback。"""

from collections.abc import Sequence
import json
import re
from urllib import error as urllib_error
from urllib import request as urllib_request

from langchain_core.messages import BaseMessage

from app.config import get_deepseek_api_key
from app.rag.prompt import ABSTAIN_ANSWER


# 增加于阶段 9.4：定义 DeepSeek OpenAI 兼容接口的默认连接参数。
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 30.0


# 增加于阶段 9.4：把 LangChain 消息转换为 DeepSeek API 消息 JSON。
def _serialize_message(message: BaseMessage) -> dict[str, str]:
    """将单条 LangChain 消息转换为 DeepSeek 兼容的 role/content 字典。

    实现方式：根据 LangChain 消息类型映射 system、human、assistant 角色，并要求
    内容为字符串；该函数只做协议格式转换，不发送网络请求或修改消息对象。

    参数：
        message: LangChain BaseMessage，通常来自 ChatPromptValue.to_messages()。

    返回：
        dict[str, str]：包含 API 所需 role 和 content 字段的字典。

    异常：
        TypeError: message 不是 BaseMessage，或消息内容不是字符串时抛出。
    """
    if not isinstance(message, BaseMessage):
        raise TypeError("messages must contain LangChain BaseMessage objects")
    if not isinstance(message.content, str):
        raise TypeError("message content must be a string")
    role = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }.get(message.type, "user")
    return {"role": role, "content": message.content}


# 增加于阶段 9.4：封装 DeepSeek OpenAI 兼容聊天接口。
class DeepSeekChatModel:
    """通过标准库 HTTP 客户端调用 DeepSeek Chat Completions。"""

    # 增加于阶段 9.4：初始化 DeepSeek 客户端配置。
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    ) -> None:
        """初始化 DeepSeek 模型连接参数。

        实现方式：优先使用显式 api_key，否则读取项目根目录 `.env` 中的
        DEEPSEEK_API_KEY；校验接口地址、
        模型名和超时后保存配置。初始化阶段不发起网络请求，便于测试和启动检查。

        参数：
            api_key: DeepSeek API Key，可选；未提供时读取环境变量。
            base_url: OpenAI 兼容 Chat Completions 地址，必须为非空字符串。
            model: 要调用的 DeepSeek 模型名，必须为非空字符串。
            timeout: 单次 HTTP 请求超时时间，单位为秒，必须为正数。

        返回：
            无返回值；实例保存后续 invoke() 所需配置。

        异常：
            TypeError: 参数类型不符合要求时抛出。
            ValueError: 地址、模型名为空或超时时间非正数时抛出。
        """
        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a string or None")
        if not isinstance(base_url, str):
            raise TypeError("base_url must be a string")
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        resolved_api_key = api_key if api_key is not None else get_deepseek_api_key()
        self.api_key = resolved_api_key.strip() if resolved_api_key else None
        self.base_url = base_url.strip()
        self.model = model.strip()
        self.timeout = float(timeout)

    # 增加于阶段 9.4：向 DeepSeek 发送 LangChain 消息并提取文本回答。
    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        """调用 DeepSeek 并返回模型生成的文本回答。

        实现方式：校验消息列表和 API Key，将 LangChain 消息转换为 OpenAI 兼容 JSON，
        通过标准库 urllib POST 请求 DeepSeek，再解析 choices[0].message.content。
        API Key 不会写入异常文本或标准输出。

        参数：
            messages: 按对话顺序排列的 LangChain BaseMessage 序列，不能为空。

        返回：
            str：DeepSeek 返回的非空回答文本。

        异常：
            TypeError: messages 不是消息序列或元素类型/内容不正确时抛出。
            RuntimeError: 未配置 API Key、网络请求失败、响应不是合法 JSON 或响应缺少回答时抛出。
        """
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of BaseMessage")
        message_list = list(messages)
        if not message_list:
            raise ValueError("messages must not be empty")
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek")

        payload = {
            "model": self.model,
            "messages": [_serialize_message(message) for message in message_list],
            "temperature": 0,
        }
        request = urllib_request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(
                f"DeepSeek API 请求失败：HTTP {error.code}，{detail}"
            ) from error
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"DeepSeek API 请求或响应解析失败：{error}") from error

        try:
            answer = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek 响应缺少 choices[0].message.content") from error
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("DeepSeek 返回了空回答")
        return answer.strip()


# 增加于阶段 9.4：提供无外部 API Key 时的可重复 grounded fallback。
class GroundedFallbackChatModel:
    """从输入资料提取 V1 演示答案，不调用外部模型或补充资料外知识。"""

    # 增加于阶段 9.4：根据资料消息生成最小的本地回答。
    # 修改于阶段 13.2：资料不足时输出需求文档规定的固定拒答话术。
    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        """从消息中的资料提取上海住宿金额，否则返回固定拒答话术。

        实现方式：拼接消息文本，仅识别资料中明确出现的“普通员工 + 金额/晚”模式；
        匹配成功时原样复述金额，其他问题统一返回不知道，确保 fallback 不会虚构
        企业政策。该实现只用于无 API Key 的 Demo 和测试，不替代真实 DeepSeek；
        无法从资料中找到答案时返回阶段 13.2 规定的 ABSTAIN_ANSWER。

        参数：
            messages: LangChain BaseMessage 序列，通常包含 System 和 Human 消息。

        返回：
            str：资料中识别到住宿上限时的 grounded 回答，否则返回固定拒答话术。

        异常：
            TypeError: messages 不是消息序列或元素不是 BaseMessage 时抛出。
        """
        if isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of BaseMessage")
        message_list = list(messages)
        if any(not isinstance(message, BaseMessage) for message in message_list):
            raise TypeError("messages must contain LangChain BaseMessage objects")
        content = "\n".join(
            message.content
            for message in message_list
            if isinstance(message.content, str)
        )
        match = re.search(r"普通员工\s+(\d+\s*元/晚)", content)
        if match:
            return f"根据提供资料，普通员工去上海出差的酒店最多可报销 {match.group(1)}。"
        return ABSTAIN_ANSWER


# 增加于阶段 9.4：根据运行环境选择真实 DeepSeek 或本地 fallback。
def get_default_chat_model() -> DeepSeekChatModel | GroundedFallbackChatModel:
    """创建当前环境可用的默认聊天模型。

    实现方式：存在非空 `.env` 配置 DEEPSEEK_API_KEY 时创建真实
    DeepSeekChatModel；没有密钥时返回 GroundedFallbackChatModel，保证本地 Demo
    可运行且不会伪装成远程模型。

    参数：
        无入参；读取项目根目录 `.env` 文件。

    返回：
        DeepSeekChatModel | GroundedFallbackChatModel：按环境选择的模型适配器。

    异常：
        ValueError、TypeError: 环境配置被显式设置为非法值时由模型初始化抛出。
    """
    api_key = get_deepseek_api_key()
    if api_key and api_key.strip():
        return DeepSeekChatModel(api_key=api_key)
    return GroundedFallbackChatModel()
