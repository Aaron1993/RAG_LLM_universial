"""空实现:直接按原始检索分数截断,不做重排。"""

from __future__ import annotations

from ..vectorstore.base import RetrievedChunk
from .base import Reranker


class NoOpReranker(Reranker):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]:
        return chunks[:top_n]
