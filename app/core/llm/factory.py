"""LLM 工厂:按配置构建 Provider。"""

from __future__ import annotations

from ...config import Settings
from .base import LLMProvider
from .openai_compat import OpenAICompatLLM


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai_compat":
        return OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"未知的 LLM provider: {settings.llm_provider}")
