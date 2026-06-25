"""RAG 高层编排:把检索、会话记忆、LLM 调用、引用生成串成一次问答。

这是整个服务的业务核心。两条主链路:
- answer():非流式,一次性返回完整答案 + 引用。
- answer_stream():流式,逐 token 产出,末尾再给出引用与会话 ID。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Sequence

from ..config import Settings
from ..core.llm.base import LLMProvider
from ..core.retrieval.retriever import Retriever
from ..core.vectorstore.base import RetrievedChunk
from ..memory.base import ConversationStore, StoredMessage
from ..observability.logging import get_logger, log_context, log_event
from ..prompts.templates import build_messages

logger = get_logger("app.rag")

# 引用片段回显的最大字符数
_SNIPPET_LEN = 200


@dataclass
class Citation:
    """一条引用来源,index 与回答中的 [编号] 对应。"""

    index: int
    document_id: str | None
    source: str | None
    page: int | None
    score: float
    snippet: str


@dataclass
class ChatResult:
    answer: str
    citations: list[Citation]
    conversation_id: str


class RAGService:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: Retriever,
        memory: ConversationStore,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._memory = memory
        self._settings = settings

    @staticmethod
    def _build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
        """把检索片段转换为引用列表,编号与提示词中的 [n] 一一对应。"""
        citations: list[Citation] = []
        for index, chunk in enumerate(chunks, start=1):
            citations.append(
                Citation(
                    index=index,
                    document_id=chunk.metadata.get("document_id"),
                    source=chunk.metadata.get("source"),
                    page=chunk.metadata.get("page"),
                    score=round(chunk.score, 4),
                    snippet=chunk.text[:_SNIPPET_LEN],
                )
            )
        return citations

    @property
    def _history_limit(self) -> int:
        # 每轮包含 user + assistant 两条消息
        return self._settings.max_history_turns * 2

    async def answer(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        # 0. 解析会话 ID(不传则新建),并把它绑定到本次调用的所有日志
        conversation_id = conversation_id or uuid.uuid4().hex
        with log_context(conversation_id=conversation_id):
            started = time.perf_counter()

            # 1. 载入历史对话(用于多轮上下文)
            history = await self._memory.load(conversation_id, limit=self._history_limit)

            # 2. 检索相关片段(向量召回 +(可选)重排)
            chunks = await self._retriever.retrieve(query, top_k=top_k, filters=filters)

            # 3. 组装提示词(system + 历史 + 带编号上下文的当前问题)
            messages = build_messages(query, chunks, history)

            # 4. 调用大模型生成回答
            answer = await self._llm.chat(messages)

            # 5. 落库本轮对话,供后续轮次使用
            await self._memory.append(
                conversation_id,
                [
                    StoredMessage(role="user", content=query),
                    StoredMessage(role="assistant", content=answer),
                ],
            )

            log_event(
                logger,
                logging.INFO,
                "chat_answered",
                retrieved=len(chunks),
                answer_len=len(answer),
                cost_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            # 6. 返回回答 + 引用(编号→来源)
            return ChatResult(
                answer=answer,
                citations=self._build_citations(chunks),
                conversation_id=conversation_id,
            )

    async def answer_stream(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[dict]:
        """流式问答,依次产出事件:
        {"type": "token", "data": "增量文本"}
        {"type": "citations", "data": [Citation...]}
        {"type": "done", "data": {"conversation_id": "..."}}
        """
        conversation_id = conversation_id or uuid.uuid4().hex
        with log_context(conversation_id=conversation_id):
            started = time.perf_counter()

            # 1~3. 与 answer() 一致:载入历史 → 检索 → 组装提示词
            history = await self._memory.load(conversation_id, limit=self._history_limit)
            chunks = await self._retriever.retrieve(query, top_k=top_k, filters=filters)
            messages = build_messages(query, chunks, history)

            # 4. 流式调用 LLM,边收边吐 token,同时累积完整答案用于落库
            parts: list[str] = []
            async for token in self._llm.chat_stream(messages):
                parts.append(token)
                yield {"type": "token", "data": token}

            answer = "".join(parts)
            # 5. 落库本轮对话
            await self._memory.append(
                conversation_id,
                [
                    StoredMessage(role="user", content=query),
                    StoredMessage(role="assistant", content=answer),
                ],
            )

            log_event(
                logger,
                logging.INFO,
                "chat_streamed",
                retrieved=len(chunks),
                answer_len=len(answer),
                cost_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            # 6. 末尾补发引用与会话 ID
            yield {"type": "citations", "data": [asdict(c) for c in self._build_citations(chunks)]}
            yield {"type": "done", "data": {"conversation_id": conversation_id}}
