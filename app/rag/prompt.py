"""阶段 9.3：企业知识库问答的 System Prompt 和消息模板。"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue


# 增加于阶段 9.3：定义限制模型回答边界的固定系统规则。
SYSTEM_PROMPT = """你是企业知识库问答助手。

只能根据提供资料回答。
没有答案就说不知道。
禁止根据一般常识补充公司内部政策。
涉及数字必须来自资料。
不要虚构来源。"""


# 增加于阶段 9.3：定义 LangChain System/Human 消息模板。
_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "请根据以下资料回答问题。\n\n资料：\n{context}\n\n问题：\n{question}",
        ),
    ]
)


# 增加于阶段 9.3：将问题和 Context 渲染为标准 ChatPromptValue。
def build_chat_prompt(question: str, context: str) -> ChatPromptValue:
    """把用户问题和检索 Context 渲染为 System/Human 消息。

    实现方式：校验问题和 Context 均为非空字符串，去除输入两端空白后交给
    LangChain ChatPromptTemplate，返回包含固定 System Prompt 和用户 Human 消息的
    ChatPromptValue；本函数不调用模型，也不修改原始检索对象。

    参数：
        question: 用户自然语言问题，必须是非空字符串。
        context: 由检索 Chunk 拼接出的资料 Context，必须是非空字符串。

    返回：
        ChatPromptValue：可直接转换为消息列表并传给聊天模型的 Prompt 值。

    异常：
        TypeError: question 或 context 不是字符串时抛出。
        ValueError: question 或 context 为空或只包含空白时抛出。
    """
    if not isinstance(question, str) or not isinstance(context, str):
        raise TypeError("question and context must be strings")
    if not question.strip():
        raise ValueError("question must not be empty")
    if not context.strip():
        raise ValueError("context must not be empty")

    return _CHAT_PROMPT.invoke(
        {"question": question.strip(), "context": context.strip()}
    )
