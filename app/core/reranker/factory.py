"""Reranker 工厂。默认关闭(NoOp)。"""

from __future__ import annotations

from ...config import Settings
from .base import Reranker
from .dashscope_reranker import DashScopeReranker
from .noop import NoOpReranker


def build_reranker(settings: Settings) -> Reranker:
    if not settings.reranker_enabled or settings.reranker_provider == "noop":
        return NoOpReranker()
    if settings.reranker_provider == "dashscope":
        return DashScopeReranker(
            api_key=settings.dashscope_api_key,
            model=settings.reranker_model,
        )
    raise ValueError(f"未知的 reranker provider: {settings.reranker_provider}")
