"""Embedding 工厂。"""

from __future__ import annotations

from ...config import Settings
from .base import EmbeddingProvider
from .openai_compat import OpenAICompatEmbedding


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai_compat":
        return OpenAICompatEmbedding(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
    raise ValueError(f"未知的 Embedding provider: {settings.embedding_provider}")
