"""向量库工厂:在 Qdrant 与 Redis Stack 之间按配置切换。"""

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
    if settings.vector_store == "redis":
        # 延迟导入,避免未用 redis 时强依赖 redis-py
        from .redis_store import RedisStackVectorStore

        return RedisStackVectorStore(
            url=settings.redis_vector_url,
            index_name=settings.redis_index_name,
            prefix=settings.redis_vector_prefix,
            tags=settings.redis_index_tags,
        )
    raise ValueError(f"未知的向量库: {settings.vector_store}")
