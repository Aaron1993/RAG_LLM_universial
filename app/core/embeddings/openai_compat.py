"""基于 OpenAI 兼容协议的 Embedding 实现(阿里百炼 text-embedding-v3 等)。"""

from __future__ import annotations

from typing import Sequence

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...exceptions import RateLimitedError, UpstreamError
from .base import EmbeddingProvider

_RETRYABLE = (APIConnectionError, APITimeoutError)


class OpenAICompatEmbedding(EmbeddingProvider):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        batch_size: int = 16,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=0,
        )
        self._model = model
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._max_retries = max_retries

    @property
    def dim(self) -> int:
        return self._dim

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(_RETRYABLE),
            ):
                with attempt:
                    resp = await self._client.embeddings.create(model=self._model, input=list(batch))
                    return [item.embedding for item in resp.data]
        except OpenAIRateLimitError as exc:
            raise RateLimitedError("Embedding 服务限流,请稍后重试") from exc
        except APIError as exc:
            raise UpstreamError(f"Embedding 上游错误: {exc}") from exc
        except _RETRYABLE as exc:
            raise UpstreamError(f"Embedding 连接失败: {exc}") from exc
        return []  # pragma: no cover

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # 按 batch_size 分批请求,避免单次输入过大被服务端拒绝;顺序拼接保证与入参一一对应
        texts = list(texts)
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[i : i + self._batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed_batch([text])
        return result[0]

    async def close(self) -> None:
        await self._client.close()
