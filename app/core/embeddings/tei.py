"""本地 Embedding 实现:HuggingFace text-embeddings-inference (TEI) HTTP 服务。

适用于本地/私有化部署的开源向量模型,如 BAAI/bge-large-zh-v1.5(维度 1024)。
调用 TEI 原生 `POST /embed` 接口:
    请求  {"inputs": ["文本1", "文本2"], "normalize": true, "truncate": true}
    响应  [[...], [...]]   # 与 inputs 顺序一一对应的向量数组

部署示例(GPU):
    docker run --gpus all -p 8080:80 \
        ghcr.io/huggingface/text-embeddings-inference:latest \
        --model-id BAAI/bge-large-zh-v1.5
然后在 .env 中设置 EMBEDDING_PROVIDER=tei、TEI_URL=http://localhost:8080、EMBEDDING_DIM=1024。
"""

from __future__ import annotations

from typing import Sequence

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...exceptions import RateLimitedError, UpstreamError
from .base import EmbeddingProvider


class TEIEmbedding(EmbeddingProvider):
    def __init__(
        self,
        *,
        base_url: str,
        dim: int,
        api_key: str = "",
        batch_size: int = 16,
        timeout: float = 30.0,
        max_retries: int = 2,
        normalize: bool = True,
        truncate: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._dim = dim
        self._batch_size = max(1, batch_size)
        self._max_retries = max_retries
        self._normalize = normalize
        # truncate=True:输入超过模型最大长度时由服务端截断,避免 413 报错
        self._truncate = truncate

        # 允许注入 client(便于测试);否则自建,并负责其生命周期
        self._owns_client = client is None
        if client is None:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)
        self._client = client

    @property
    def dim(self) -> int:
        return self._dim

    async def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        payload = {"inputs": list(batch), "normalize": self._normalize, "truncate": self._truncate}
        try:
            # 仅对连接/超时类错误重试(指数退避);HTTP 4xx/5xx 不重试
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._max_retries + 1),
                wait=wait_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(httpx.TransportError),
            ):
                with attempt:
                    resp = await self._client.post("/embed", json=payload)
                    resp.raise_for_status()
                    return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise RateLimitedError("TEI 服务限流,请稍后重试") from exc
            raise UpstreamError(f"TEI 上游错误: HTTP {exc.response.status_code}") from exc
        except httpx.TransportError as exc:  # 重试耗尽
            raise UpstreamError(f"TEI 连接失败: {exc}") from exc
        return []  # pragma: no cover

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # 按 batch_size 分批,顺序拼接以保证与入参一一对应
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
        if self._owns_client:
            await self._client.aclose()
