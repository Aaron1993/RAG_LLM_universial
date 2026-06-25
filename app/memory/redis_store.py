"""基于 Redis 的会话记忆(多实例部署可选,需 pip install redis)。"""

from __future__ import annotations

import json

from .base import ConversationStore, StoredMessage

_KEY_PREFIX = "rag:conv:"
# 单会话最多保留的消息条数(防止无限增长)
_MAX_KEEP = 200


class RedisConversationStore(ConversationStore):
    def __init__(self, url: str) -> None:
        try:
            from redis.asyncio import from_url
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("使用 redis 记忆后端需先安装: pip install redis") from exc
        self._client = from_url(url, encoding="utf-8", decode_responses=True)

    def _key(self, conversation_id: str) -> str:
        return f"{_KEY_PREFIX}{conversation_id}"

    async def load(self, conversation_id: str, *, limit: int | None = None) -> list[StoredMessage]:
        key = self._key(conversation_id)
        if limit is not None and limit >= 0:
            raw = await self._client.lrange(key, -limit, -1)
        else:
            raw = await self._client.lrange(key, 0, -1)
        messages = []
        for item in raw:
            data = json.loads(item)
            messages.append(StoredMessage(role=data["role"], content=data["content"]))
        return messages

    async def append(self, conversation_id: str, messages: list[StoredMessage]) -> None:
        if not messages:
            return
        key = self._key(conversation_id)
        payloads = [json.dumps({"role": m.role, "content": m.content}, ensure_ascii=False) for m in messages]
        await self._client.rpush(key, *payloads)
        await self._client.ltrim(key, -_MAX_KEEP, -1)

    async def clear(self, conversation_id: str) -> None:
        await self._client.delete(self._key(conversation_id))

    async def close(self) -> None:
        await self._client.aclose()
