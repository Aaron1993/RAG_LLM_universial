"""向量库抽象接口与数据结构。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VectorRecord:
    """写入向量库的一条记录。"""

    id: str
    vector: list[float]
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """检索命中的片段。"""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dim: int) -> None:
        """确保集合存在(不存在则创建)。"""
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, *, ids: list[str] | None = None, filters: dict | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> bool:
        """连通性探测,用于 /readyz。"""
        raise NotImplementedError

    async def close(self) -> None:  # 默认无操作
        return None
