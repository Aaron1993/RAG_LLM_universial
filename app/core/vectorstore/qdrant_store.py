"""Qdrant 向量库实现。"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from ...exceptions import UpstreamError
from ...observability.logging import get_logger
from .base import RetrievedChunk, VectorRecord, VectorStore

logger = get_logger("app.vectorstore")

# payload 中保存正文的字段名
_TEXT_KEY = "text"


def _build_filter(filters: dict | None) -> models.Filter | None:
    if not filters:
        return None
    must = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return models.Filter(must=must)


class QdrantVectorStore(VectorStore):
    def __init__(self, *, url: str, api_key: str, collection: str) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)
        self._collection = collection

    async def ensure_collection(self, dim: int) -> None:
        try:
            existing = await self._client.get_collections()
            names = {c.name for c in existing.collections}
            if self._collection not in names:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )
                logger.info("collection_created", extra={"extra_fields": {"collection": self._collection, "dim": dim}})
        except Exception as exc:  # 启动期连接失败应快速暴露
            raise UpstreamError(f"Qdrant 初始化失败: {exc}") from exc

    async def upsert(self, records: list[VectorRecord]) -> None:
        # 正文存在 payload 的 _TEXT_KEY 字段,其余元数据平铺到 payload,便于按字段过滤/删除
        if not records:
            return
        points = [
            models.PointStruct(
                id=record.id,
                vector=record.vector,
                payload={_TEXT_KEY: record.text, **record.metadata},
            )
            for record in records
        ]
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise UpstreamError(f"Qdrant 写入失败: {exc}") from exc

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                query_filter=_build_filter(filters),
                score_threshold=score_threshold or None,
                with_payload=True,
            )
        except Exception as exc:
            raise UpstreamError(f"Qdrant 检索失败: {exc}") from exc

        # 把命中点还原为 RetrievedChunk:正文从 payload 取出,剩余字段即元数据
        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop(_TEXT_KEY, "")
            chunks.append(
                RetrievedChunk(id=str(point.id), text=text, score=float(point.score), metadata=payload)
            )
        return chunks

    async def delete(self, *, ids: list[str] | None = None, filters: dict | None = None) -> None:
        try:
            if ids:
                await self._client.delete(
                    collection_name=self._collection,
                    points_selector=models.PointIdsList(points=ids),
                )
            if filters:
                await self._client.delete(
                    collection_name=self._collection,
                    points_selector=models.FilterSelector(filter=_build_filter(filters)),
                )
        except Exception as exc:
            raise UpstreamError(f"Qdrant 删除失败: {exc}") from exc

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
