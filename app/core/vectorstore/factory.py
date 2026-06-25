"""向量库工厂。"""

from __future__ import annotations

from ...config import Settings
from .base import VectorStore
from .qdrant_store import QdrantVectorStore


def build_vectorstore(settings: Settings) -> VectorStore:
    if settings.vector_store == "qdrant":
        return QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
        )
    raise ValueError(f"未知的向量库: {settings.vector_store}")
