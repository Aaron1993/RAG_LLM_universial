"""会话记忆抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredMessage:
    role: str  # user | assistant
    content: str


class ConversationStore(ABC):
    @abstractmethod
    async def load(self, conversation_id: str, *, limit: int | None = None) -> list[StoredMessage]:
        """返回最近 limit 条消息(按时间正序)。"""
        raise NotImplementedError

    @abstractmethod
    async def append(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def clear(self, conversation_id: str) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None
