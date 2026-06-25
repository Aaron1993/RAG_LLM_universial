"""会话记忆工厂。"""

from __future__ import annotations

from ..config import Settings
from .base import ConversationStore
from .memory_store import InMemoryConversationStore
from .sqlite_store import SqliteConversationStore


def build_memory(settings: Settings) -> ConversationStore:
    if settings.memory_backend == "sqlite":
        return SqliteConversationStore(settings.sqlite_path)
    if settings.memory_backend == "memory":
        return InMemoryConversationStore()
    if settings.memory_backend == "redis":
        from .redis_store import RedisConversationStore

        return RedisConversationStore(settings.redis_url)
    raise ValueError(f"未知的记忆后端: {settings.memory_backend}")
