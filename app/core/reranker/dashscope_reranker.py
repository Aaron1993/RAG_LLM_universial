"""阿里百炼 DashScope 文本重排实现(gte-rerank-v2)。

通过 DashScope 原生 rerank HTTP 接口调用(OpenAI 兼容协议不含 rerank)。
为保证可用性:重排失败时降级为原始顺序,仅记录告警,不阻断主流程。
"""

from __future__ import annotations

import httpx

from ...observability.logging import get_logger
from ..vectorstore.base import RetrievedChunk
from .base import Reranker

logger = get_logger("app.reranker")

_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class DashScopeReranker(Reranker):
    def __init__(self, *, api_key: str, model: str = "gte-rerank-v2", timeout: float = 15.0) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        payload = {
            "model": self._model,
            "input": {"query": query, "documents": [c.text for c in chunks]},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(_RERANK_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            results = data["output"]["results"]
        except Exception as exc:  # 降级:不阻断问答
            logger.warning("rerank_failed_fallback", extra={"extra_fields": {"error": str(exc)}})
            return chunks[:top_n]

        reranked: list[RetrievedChunk] = []
        for item in results:
            idx = item["index"]
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                chunk.score = float(item.get("relevance_score", chunk.score))
                reranked.append(chunk)
        return reranked[:top_n] if reranked else chunks[:top_n]
