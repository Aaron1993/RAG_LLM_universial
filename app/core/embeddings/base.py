"""Embedding 提供方抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度,需与向量库集合维度一致。"""
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化文档片段。"""
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """向量化单条查询。"""
        raise NotImplementedError

    async def close(self) -> None:
        """释放底层资源(如本地 HTTP 连接池);默认无操作。"""
        return None
