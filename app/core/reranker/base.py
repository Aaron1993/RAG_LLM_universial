"""重排序抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..vectorstore.base import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError
