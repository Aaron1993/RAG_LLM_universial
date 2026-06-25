"""进程内会话记忆(开发 / 测试用,重启即失)。"""

from __future__ import annotations

from collections import defaultdict

from .base import ConversationStore, StoredMessage


class InMemoryConversationStore(ConversationStore):
    def __init__(self) -> None:
        self._data: dict[str, list[StoredMessage]] = defaultdict(list)

    async def load(self, conversation_id: str, *, limit: int | None = None) -> list[StoredMessage]:
        messages = self._data.get(conversation_id, [])
        if limit is not None and limit >= 0:
            return list(messages[-limit:])
        return list(messages)

    async def append(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        self._data[conversation_id].extend(messages)

    async def clear(self, conversation_id: str) -> None:
        self._data.pop(conversation_id, None)
