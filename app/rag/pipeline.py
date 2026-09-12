"""阶段 9.4–9.5：串接 Retriever、Context、Prompt 和聊天模型。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from app.rag.context_builder import MAX_CONTEXT_CHUNKS, build_context, limit_chunks
from app.rag.llm import get_default_chat_model
from app.rag.prompt import build_chat_prompt
from app.retrieval.vector_retriever import RetrievedChunk, search
from app.security import User


# 增加于阶段 9.4：定义可注入 Retriever 和模型的最小调用类型。
Retriever = Callable[..., list[RetrievedChunk]]


# 增加于阶段 9.4：定义 Pipeline 的结构化输出。
@dataclass(frozen=True)
class RagResult:
    """保存 RAG 回答、实际 Context 和最终保留的检索 Chunk。"""

    answer: str
    context: str
    chunks: tuple[RetrievedChunk, ...]


# 增加于阶段 9.4：实现 Query 到回答的五段式 RAG Pipeline。
class RagPipeline:
    """使用 LangChain Runnable 串联检索、Context、Prompt 和聊天模型。"""

    # 增加于阶段 9.4：初始化 Pipeline 的依赖和 Runnable 阶段。
    # 修改于阶段 12.3：将 User 传入默认 Retriever，确保检索前应用权限过滤。
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: object | None = None,
        top_k: int = MAX_CONTEXT_CHUNKS,
        user: User | None = None,
    ) -> None:
        """初始化一个可使用真实依赖或测试替身的 RAG Pipeline。

        实现方式：校验 top_k 和可选 User，默认使用阶段 8 的 search() 和按环境选择
        的聊天模型，再为 Retriever、数量控制、Context、Prompt 和 LLM 分别创建
        RunnableLambda。使用默认 Retriever 时，User 会传入 search() 触发 Qdrant
        权限过滤；依赖可由调用方注入，便于离线测试而不改变正式数据流。

        参数：
            retriever: 接收 query 和 top_k 关键字参数并返回 RetrievedChunk 列表的函数。
            llm: 接收 LangChain BaseMessage 序列并返回字符串的对象或可调用对象。
            top_k: Retriever 请求的候选数量，必须为正整数；最终 Context 仍最多 5 个 Chunk。
            user: 当前请求用户；使用默认 Qdrant Retriever 时必须提供，注入自定义
                Retriever 时可省略以兼容离线测试。

        返回：
            无返回值；实例保存可执行的 RAG Runnable 阶段。

        异常：
            TypeError: top_k 不是整数时抛出。
            ValueError: top_k 小于等于 0 时抛出。
            TypeError: user 不是 User 对象时抛出。
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if user is not None and not isinstance(user, User):
            raise TypeError("user must be a User")

        self.retriever = retriever or search
        self.llm = llm or get_default_chat_model()
        self.top_k = top_k
        self.user = user
        self._retrieve = RunnableLambda(
            lambda query: self.retriever(query, top_k=self.top_k)
            if self.user is None
            else self.retriever(query, top_k=self.top_k, user=self.user)
        )
        self._limit = RunnableLambda(
            lambda chunks: limit_chunks(chunks, max_chunks=MAX_CONTEXT_CHUNKS)
        )
        self._build_context = RunnableLambda(build_context)
        self._build_prompt = RunnableLambda(
            lambda values: build_chat_prompt(values["query"], values["context"])
        )
        self._invoke_llm = RunnableLambda(self._invoke_model)

    # 增加于阶段 9.4：兼容带 invoke 方法和普通可调用对象的模型。
    def _invoke_model(self, prompt_value: ChatPromptValue) -> str:
        """把 Prompt 消息交给注入聊天模型并校验文本回答。

        实现方式：将 ChatPromptValue 转为消息列表，优先调用模型对象的 invoke 方法，
        否则调用普通 callable；最后校验返回值为非空字符串，防止空回答继续流向用户。

        参数：
            prompt_value: 已由 build_chat_prompt() 渲染的 LangChain Prompt 值。

        返回：
            str：模型返回的去除首尾空白的回答。

        异常：
            TypeError: 模型没有 invoke 方法且不可调用，或返回值不是字符串时抛出。
            RuntimeError: 模型返回空字符串时抛出。
        """
        messages: Sequence[BaseMessage] = prompt_value.to_messages()
        invoke = getattr(self.llm, "invoke", None)
        if callable(invoke):
            answer = invoke(messages)
        elif callable(self.llm):
            answer = self.llm(messages)
        else:
            raise TypeError("llm must be callable or provide invoke(messages)")
        if not isinstance(answer, str):
            raise TypeError("llm answer must be a string")
        if not answer.strip():
            raise RuntimeError("llm returned an empty answer")
        return answer.strip()

    # 增加于阶段 9.4：执行完整 RAG 流程并返回结构化结果。
    def invoke(self, query: str) -> RagResult:
        """执行 Query → Retriever → Context → Prompt → LLM 的完整流程。

        实现方式：校验 Query 后依次调用五个 Runnable；检索结果先按相关度顺序限制
        为最多五个，再拼接 Context、渲染 Chat Prompt 并调用模型。没有召回资料时直接
        返回“根据提供资料无法确定”，避免把空 Context 发送给模型。

        参数：
            query: 用户自然语言问题，必须是非空字符串。

        返回：
            RagResult：包含最终回答、实际送入模型的 Context 和保留的 Chunk 元组。

        异常：
            TypeError: query 不是字符串，或依赖返回的数据类型不正确时抛出。
            ValueError: query 为空时抛出。
            RuntimeError: 注入模型返回空回答，或真实模型请求失败时抛出。
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")

        retrieved = self._retrieve.invoke(query.strip())
        retained = self._limit.invoke(retrieved)
        if not retained:
            return RagResult(
                answer="根据提供资料无法确定。",
                context="",
                chunks=(),
            )

        context = self._build_context.invoke(retained)
        prompt_value = self._build_prompt.invoke(
            {"query": query.strip(), "context": context}
        )
        answer = self._invoke_llm.invoke(prompt_value)
        return RagResult(answer=answer, context=context, chunks=tuple(retained))
