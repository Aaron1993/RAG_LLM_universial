"""检索编排:query 向量化 -> 向量召回 -> (可选)重排。"""

from __future__ import annotations

import logging

from ...config import Settings
from ...observability.logging import get_logger, log_event
from ..embeddings.base import EmbeddingProvider
from ..reranker.base import Reranker
from ..vectorstore.base import RetrievedChunk, VectorStore

logger = get_logger("app.retrieval")


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vectorstore: VectorStore,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._reranker = reranker
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """返回与 query 最相关的若干片段。

        流程:
        1. 计算最终返回条数 k 与候选召回条数 fetch_k(开启重排时多召回再精排);
        2. 将 query 向量化;
        3. 在向量库中按相似度召回候选(可带元数据过滤与分数阈值);
        4. 若开启重排,用 reranker 精排并截断到 k;否则直接截断。
        """
        # 1. 确定返回条数;开启重排时,先多召回候选以提升精排上限
        k = top_k or self._settings.retrieval_top_k
        fetch_k = max(k * 4, self._settings.rerank_top_n) if self._settings.reranker_enabled else k

        # 2. query 向量化
        query_vector = await self._embeddings.embed_query(query)

        # 3. 向量召回
        candidates = await self._vectorstore.search(
            query_vector,
            top_k=fetch_k,
            filters=filters,
            score_threshold=self._settings.score_threshold,
        )
        if not candidates:
            log_event(logger, logging.DEBUG, "retrieve_empty", fetch_k=fetch_k)
            return []

        # 4. 重排(NoOp 时等价于按召回分数截断)
        if self._settings.reranker_enabled:
            result = await self._reranker.rerank(query, candidates, top_n=k)
        else:
            result = candidates[:k]

        log_event(
            logger,
            logging.DEBUG,
            "retrieve_done",
            candidates=len(candidates),
            returned=len(result),
            reranked=self._settings.reranker_enabled,
        )
        return result
