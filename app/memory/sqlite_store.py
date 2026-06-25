"""基于 SQLite 的会话记忆(单机默认实现,持久化)。"""

from __future__ import annotations

import os

import aiosqlite

from .base import ConversationStore, StoredMessage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


class SqliteConversationStore(ConversationStore):
    def __init__(self, path: str) -> None:
        self._path = path
        self._ready = False
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._ready = True

    async def load(self, conversation_id: str, *, limit: int | None = None) -> list[StoredMessage]:
        await self._ensure()
        async with aiosqlite.connect(self._path) as db:
            if limit is not None and limit >= 0:
                cursor = await db.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (conversation_id, limit),
                )
                rows = await cursor.fetchall()
                rows = list(reversed(rows))
            else:
                cursor = await db.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                    (conversation_id,),
                )
                rows = await cursor.fetchall()
        return [StoredMessage(role=row[0], content=row[1]) for row in rows]

    async def append(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        if not messages:
            return
        await self._ensure()
        async with aiosqlite.connect(self._path) as db:
            await db.executemany(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                [(conversation_id, m.role, m.content) for m in messages],
            )
            await db.commit()

    async def clear(self, conversation_id: str) -> None:
        await self._ensure()
        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            await db.commit()
