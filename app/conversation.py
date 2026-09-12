"""阶段 11.1–11.4：生成、保存、限制并解析 Conversation History。"""

# 修改于阶段 11.3：增加 6～10 轮的历史长度控制边界。
# 修改于阶段 11.4：按用户要求暂不限制上下文长度，完整读取会话历史。

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Literal
from uuid import uuid4

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, select


# 增加于阶段 11.2：定义 SQLAlchemy Core 的会话消息表，兼容 SQLite 和 PostgreSQL。
DEFAULT_CONVERSATION_DATABASE_URL = "sqlite+pysqlite:///:memory:"
_CONVERSATION_METADATA = MetaData()
_CONVERSATION_MESSAGES = Table(
    "conversation_messages",
    _CONVERSATION_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(128), nullable=False, index=True),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
)


# 增加于阶段 11.1：为每个新会话生成唯一且可读的会话标识。
def create_session_id() -> str:
    """为新会话生成稳定格式的唯一 session_id。

    实现方式：使用标准库 uuid4 生成随机 UUID，取其不含连字符的十六进制形式，
    再添加 `session_` 前缀。函数只负责创建会话身份，不保存会话历史；后续阶段可将
    返回值作为数据库会话记录和 Conversation History 的关联键。

    参数：
        无入参。

    返回：
        str：格式为 `session_<32 位小写十六进制字符>` 的新会话标识。

    异常：
        无主动抛出的异常；uuid4 失败时会透传标准库产生的异常。
    """
    return f"session_{uuid4().hex}"


# 增加于阶段 11.2：定义可传递到后续 LangChain Context 的标准消息对象。
@dataclass(frozen=True)
class ConversationMessage:
    """表示一条已保存的 user 或 assistant 会话消息。"""

    role: Literal["user", "assistant"]
    content: str


# 增加于阶段 11.2：校验会话标识和消息正文等必须为非空文本。
def _normalize_required_text(value: str, field_name: str) -> str:
    """校验并清理会话存储中的必填文本字段。

    实现方式：确认输入是字符串且去除首尾空白后不为空，返回清理后的文本，统一
    处理 session_id 和消息正文的边界校验，避免非法值进入数据库。

    参数：
        value: 待校验的文本值，必须是字符串且不能只包含空白。
        field_name: 错误消息中使用的字段名称。

    返回：
        str：去除首尾空白后的非空文本。

    异常：
        TypeError: value 或 field_name 不是字符串时抛出。
        ValueError: value 去除首尾空白后为空时抛出。
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


# 增加于阶段 11.2：使用 SQLAlchemy 保存并按 session_id 读取 Conversation History。
class ConversationHistoryStore:
    """保存 user/assistant 消息并按会话标识隔离读取。"""

    # 增加于阶段 11.2：初始化数据库连接并创建会话消息表。
    def __init__(
        self,
        database_url: str = DEFAULT_CONVERSATION_DATABASE_URL,
    ) -> None:
        """初始化会话历史存储。

        实现方式：校验数据库 URL，使用 SQLAlchemy 创建 Engine，并创建所需表；默认
        使用 SQLite 内存库保证 Demo 和单测无需外部服务，调用方可传入 PostgreSQL
        URL 复用同一存储接口。

        参数：
            database_url: SQLAlchemy 数据库 URL，默认是 SQLite 内存数据库，不能为空。

        返回：
            无返回值；实例保存后续 append_message() 和 get_messages() 使用的 Engine。

        异常：
            TypeError: database_url 不是字符串时抛出。
            ValueError: database_url 为空时抛出。
            sqlalchemy.exc.SQLAlchemyError: 数据库连接或建表失败时由 SQLAlchemy 抛出。
        """
        self.database_url = _normalize_required_text(database_url, "database_url")
        self._engine = create_engine(self.database_url)
        _CONVERSATION_METADATA.create_all(self._engine)

    # 增加于阶段 11.2：写入一条属于指定会话的 user/assistant 消息。
    def append_message(
        self,
        session_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ) -> ConversationMessage:
        """保存一条会话消息并返回标准消息对象。

        实现方式：校验 session_id、角色和正文，使用 SQLAlchemy 事务向消息表追加一行；
        数据库自增 ID 保证同一会话的读取顺序与写入顺序一致。

        参数：
            session_id: 11.1 生成的会话标识，必须为非空字符串。
            role: 消息角色，只允许 `user` 或 `assistant`。
            content: 消息正文，去除首尾空白后必须非空。

        返回：
            ConversationMessage：已校验并保存的消息对象。

        异常：
            TypeError: session_id、role 或 content 类型不符合要求时抛出。
            ValueError: session_id 或 content 为空，或 role 不是 user/assistant 时抛出。
            sqlalchemy.exc.SQLAlchemyError: 数据库写入失败时由 SQLAlchemy 抛出。
        """
        normalized_session_id = _normalize_required_text(session_id, "session_id")
        if not isinstance(role, str):
            raise TypeError("role must be a string")
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        normalized_content = _normalize_required_text(content, "content")
        message = ConversationMessage(role=role, content=normalized_content)
        with self._engine.begin() as connection:
            connection.execute(
                _CONVERSATION_MESSAGES.insert().values(
                    session_id=normalized_session_id,
                    role=message.role,
                    content=message.content,
                )
            )
        return message

    # 增加于阶段 11.2：按会话标识读取全部历史并保持写入顺序。
    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        """读取指定会话的全部 user/assistant 消息。

        实现方式：校验 session_id，按消息自增 ID 升序查询数据库，只返回该会话的
        消息对象；不存在的会话返回空列表，不会读取其他会话数据。

        参数：
            session_id: 要读取历史的会话标识，必须为非空字符串。

        返回：
            list[ConversationMessage]：按保存顺序排列的消息列表；没有历史时为空列表。

        异常：
            TypeError: session_id 不是字符串时抛出。
            ValueError: session_id 为空时抛出。
            sqlalchemy.exc.SQLAlchemyError: 数据库读取失败时由 SQLAlchemy 抛出。
        """
        normalized_session_id = _normalize_required_text(session_id, "session_id")
        statement = (
            select(
                _CONVERSATION_MESSAGES.c.role,
                _CONVERSATION_MESSAGES.c.content,
            )
            .where(_CONVERSATION_MESSAGES.c.session_id == normalized_session_id)
            .order_by(_CONVERSATION_MESSAGES.c.id)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [
            ConversationMessage(role=row["role"], content=row["content"])
            for row in rows
        ]

    # 增加于阶段 11.3：提供会话历史读取入口。
    # 修改于阶段 11.4：取消轮数和消息数量限制，返回全部历史。
    def get_recent_messages(self, session_id: str) -> list[ConversationMessage]:
        """读取指定会话的完整历史，暂不执行轮数限制。

        实现方式：保留阶段 11.3 的方法名以兼容已有调用方，但直接复用完整历史查询
        get_messages()，不再按 6～10 轮或消息数量截断；后续重新启用限制时可在该
        入口统一增加策略。

        参数：
            session_id: 要读取历史的会话标识，必须为非空字符串。

        返回：
            list[ConversationMessage]：按写入顺序排列的全部消息；没有历史时为空列表。

        异常：
            TypeError: session_id 不是字符串时抛出。
            ValueError: session_id 为空时抛出。
            sqlalchemy.exc.SQLAlchemyError: 数据库读取失败时由 SQLAlchemy 抛出。
        """
        return self.get_messages(session_id)

    # 增加于阶段 11.2：显式释放 SQLAlchemy Engine 连接池资源。
    def close(self) -> None:
        """释放会话历史存储持有的数据库连接池资源。

        实现方式：调用 SQLAlchemy Engine.dispose() 关闭当前连接池；SQLAlchemy 的
        dispose 操作可重复执行，因此调用方可以在应用退出或测试清理中安全调用多次。

        参数：
            无入参。

        返回：
            无返回值；存储实例不再持有已打开的数据库连接。

        异常：
            sqlalchemy.exc.SQLAlchemyError: 连接池释放失败时由 SQLAlchemy 抛出。
        """
        self._engine.dispose()


# 增加于阶段 11.4：识别需求文档中的“那经理呢？”省略式追问。
_MANAGER_FOLLOW_UP_PATTERN = re.compile(r"^那\s*经理(?:\s*呢)?[？?。]?$")
_TRAVEL_POLICY_PATTERN = re.compile(
    r"(?P<city>[\u4e00-\u9fff]{2,8}?)(?:出差)?(?:普通员工)?住宿标准"
)


# 增加于阶段 11.4：根据最近 user 历史补全经理住宿标准追问。
def rewrite_follow_up_question(
    question: str,
    history: Iterable[ConversationMessage],
) -> str:
    """根据历史上下文把需求文档中的经理省略问题改写为完整问题。

    实现方式：校验当前问题和历史消息，只有当当前问题匹配“那经理呢？”且历史中
    存在包含城市和住宿标准的最近 user 问题时，才提取城市并生成“经理在城市出差
    住宿标准是多少？”；其他问题或无法解析的历史原样返回，避免凭空补充上下文。

    参数：
        question: 当前用户问题，必须是非空字符串。
        history: 按时间顺序排列的 ConversationMessage 可迭代对象，元素必须来自
            当前会话，允许为空。

    返回：
        str：解析成功时返回补全后的问题，否则返回清理后的原问题。

    异常：
        TypeError: question 不是字符串、history 不是可迭代对象或历史元素类型错误时抛出。
        ValueError: question 为空时抛出。
    """
    normalized_question = _normalize_required_text(question, "question")
    if isinstance(history, (str, bytes)):
        raise TypeError("history must be an iterable of ConversationMessage")
    try:
        messages = list(history)
    except TypeError as error:
        raise TypeError("history must be an iterable of ConversationMessage") from error
    if any(not isinstance(message, ConversationMessage) for message in messages):
        raise TypeError("history must contain only ConversationMessage objects")

    if not _MANAGER_FOLLOW_UP_PATTERN.fullmatch(normalized_question):
        return normalized_question

    for message in reversed(messages):
        if message.role != "user":
            continue
        match = _TRAVEL_POLICY_PATTERN.search(message.content)
        if match:
            return f"经理在{match.group('city')}出差住宿标准是多少？"
    return normalized_question
