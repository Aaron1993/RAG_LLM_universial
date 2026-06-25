"""LLM 提供方抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence, TypedDict


class Message(TypedDict):
    role: str  # system | user | assistant
    content: str


class LLMProvider(ABC):
    """对话大模型抽象。新增厂商只需实现本接口并在 factory 注册。"""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """一次性返回完整回答。"""
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式返回增量 token(异步生成器)。"""
        raise NotImplementedError
