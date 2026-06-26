"""Redis Stack 向量库实现(RediSearch + RedisJSON)。

做法对齐 Java 版 RagServiceImpl:
- 建索引:FT.CREATE <idx> ON JSON PREFIX 1 <prefix> SCHEMA
          $.text AS text TEXT
          $.metadata.<tag> AS <tag> TAG   (可过滤字段,默认 document_id/source/tenant)
          $.vector AS vector VECTOR HNSW 6 TYPE FLOAT32 DIM <dim> DISTANCE_METRIC COSINE
- 写入:  JSON.SET <prefix><id> $  {"text":..., "vector":[...], "metadata":{...}}
- 检索:  FT.SEARCH <idx> "<filter>=>[KNN k @vector $vec AS vector_score]"
          PARAMS 2 vec <小端 float32 字节> RETURN 2 vector_score $ SORTBY vector_score ASC DIALECT 2
          余弦相似度 score = (2 - distance) / 2,范围 [0,1](与 Qdrant 语义一致)
- 删除:  按 id 直接 DEL;按 filters 先 FT.SEARCH(NOCONTENT)取 key 再 DEL

需要 Redis Stack(含 RediSearch、RedisJSON 模块),并安装 redis-py:pip install redis
"""

from __future__ import annotations

import json
import struct

from ...exceptions import UpstreamError
from ...observability.logging import get_logger
from .base import RetrievedChunk, VectorRecord, VectorStore

logger = get_logger("app.vectorstore")


def _normalize_prefix(prefix: str) -> str:
    value = (prefix or "embedding:").strip()
    return value if value.endswith(":") else value + ":"


def _escape_tag(value: str) -> str:
    """转义 RediSearch TAG 过滤值中的特殊字符。"""
    for ch in ("\\", "{", "}", "|", ",", " ", "-", "@", ":"):
        value = value.replace(ch, "\\" + ch)
    return value


def _to_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _to_float32_bytes(vector: list[float]) -> bytes:
    """与 Java floatArrayToByteArray 一致:小端 FLOAT32 字节串。"""
    return struct.pack(f"<{len(vector)}f", *vector)


class RedisStackVectorStore(VectorStore):
    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        index_name: str = "rag_knowledge_index",
        prefix: str = "embedding:",
        tags: list[str] | None = None,
        client=None,
    ) -> None:
        self._index = index_name
        self._prefix = _normalize_prefix(prefix)
        # 可作为过滤条件的 TAG 字段(必须在建索引时声明,才能在检索/删除时按其过滤)
        self._tags = list(tags) if tags else ["document_id", "source", "tenant"]
        self._owns_client = client is None
        if client is None:
            try:
                from redis.asyncio import from_url
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("使用 redis 向量库需先安装: pip install redis") from exc
            # decode_responses=True:响应自动按 UTF-8 解码,便于解析 JSON;入参字节仍原样发送
            client = from_url(url, decode_responses=True)
        self._client = client

    # ---------- 建索引 ----------

    async def ensure_collection(self, dim: int) -> None:
        try:
            # 索引已存在则跳过
            await self._client.execute_command("FT.INFO", self._index)
            return
        except Exception:
            logger.info("redis_index_absent_creating", extra={"extra_fields": {"index": self._index}})

        args: list = [
            "FT.CREATE", self._index,
            "ON", "JSON",
            "PREFIX", "1", self._prefix,
            "SCHEMA",
            "$.text", "AS", "text", "TEXT",
        ]
        for tag in self._tags:
            args += [f"$.metadata.{tag}", "AS", tag, "TAG"]
        args += [
            "$.vector", "AS", "vector",
            "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
        ]
        try:
            await self._client.execute_command(*args)
            logger.info(
                "redis_index_created",
                extra={"extra_fields": {"index": self._index, "prefix": self._prefix, "dim": dim}},
            )
        except Exception as exc:
            raise UpstreamError(f"Redis 向量索引创建失败: {exc}") from exc

    # ---------- 写入 ----------

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for record in records:
                key = self._prefix + record.id
                # JSON 文档结构:正文 + 向量数组 + 元数据(metadata 平铺供 TAG 索引)
                doc = {
                    "text": record.text,
                    "vector": list(record.vector),
                    "metadata": record.metadata,
                }
                pipe.execute_command("JSON.SET", key, "$", json.dumps(doc, ensure_ascii=False))
            await pipe.execute()
        except Exception as exc:
            raise UpstreamError(f"Redis 向量写入失败: {exc}") from exc

    # ---------- 检索 ----------

    def _build_filter(self, filters: dict | None) -> str:
        """构造 RediSearch 前置过滤表达式;仅支持已建索引的 TAG 字段。"""
        if not filters:
            return "*"
        clauses = []
        for key, value in filters.items():
            if key not in self._tags:
                logger.warning(
                    "redis_filter_field_not_indexed",
                    extra={"extra_fields": {"field": key, "indexed": self._tags}},
                )
                continue
            clauses.append(f"@{key}:{{{_escape_tag(str(value))}}}")
        return f"({' '.join(clauses)})" if clauses else "*"

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        filters: dict | None = None,
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        filter_expr = self._build_filter(filters)
        knn_query = f"{filter_expr}=>[KNN {top_k} @vector $vec AS vector_score]"
        args = [
            "FT.SEARCH", self._index, knn_query,
            "PARAMS", "2", "vec", _to_float32_bytes(vector),
            "RETURN", "2", "vector_score", "$",
            "SORTBY", "vector_score", "ASC",
            "DIALECT", "2",
        ]
        try:
            response = await self._client.execute_command(*args)
        except Exception as exc:
            raise UpstreamError(f"Redis 向量检索失败: {exc}") from exc
        return await self._parse_search(response, score_threshold)

    async def _parse_search(self, response, score_threshold: float) -> list[RetrievedChunk]:
        # 响应形如:[count, key1, [field, value, ...], key2, [...], ...]
        if not isinstance(response, (list, tuple)) or len(response) < 1:
            return []

        results: list[RetrievedChunk] = []
        body = list(response[1:])
        for i in range(0, len(body) - 1, 2):
            key = _to_str(body[i])
            fields = body[i + 1]
            if not isinstance(fields, (list, tuple)):
                continue
            fmap: dict[str, str] = {}
            for j in range(0, len(fields) - 1, 2):
                fmap[_to_str(fields[j])] = _to_str(fields[j + 1])

            # 余弦距离 -> 相似度(与 Java 一致),范围 [0,1]
            distance = float(fmap.get("vector_score", "2"))
            score = (2 - distance) / 2
            if score < score_threshold:
                continue

            doc_json = fmap.get("$")
            if doc_json is None:
                # 个别 Redis 版本 RETURN $ 可能取不到,回退用 JSON.GET 取整文档
                doc_json = _to_str(await self._client.execute_command("JSON.GET", key))
            try:
                doc = json.loads(doc_json)
            except (TypeError, ValueError):
                doc = {}
            results.append(
                RetrievedChunk(
                    id=key,
                    text=doc.get("text", ""),
                    score=score,
                    metadata=doc.get("metadata", {}) or {},
                )
            )
        return results

    # ---------- 删除 ----------

    async def delete(self, *, ids: list[str] | None = None, filters: dict | None = None) -> None:
        try:
            if ids:
                keys = [self._prefix + i for i in ids]
                await self._client.delete(*keys)
            if filters:
                keys = await self._find_keys(filters)
                if keys:
                    await self._client.delete(*keys)
        except Exception as exc:
            raise UpstreamError(f"Redis 向量删除失败: {exc}") from exc

    async def _find_keys(self, filters: dict) -> list[str]:
        filter_expr = self._build_filter(filters)
        if filter_expr == "*":
            return []
        response = await self._client.execute_command(
            "FT.SEARCH", self._index, filter_expr,
            "NOCONTENT", "LIMIT", "0", "10000", "DIALECT", "2",
        )
        if not isinstance(response, (list, tuple)):
            return []
        # NOCONTENT 时响应为 [count, key1, key2, ...]
        return [_to_str(k) for k in response[1:]]

    # ---------- 运维 ----------

    async def health(self) -> bool:
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
