"""文档入库流水线:加载 -> 分块 -> 向量化 -> 写入向量库。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ..core.embeddings.base import EmbeddingProvider
from ..core.vectorstore.base import VectorRecord, VectorStore
from ..observability.logging import get_logger
from .loaders import LoadedSection, get_loader
from .splitter import RecursiveTextSplitter

logger = get_logger("app.ingestion")

# 用于生成稳定 point id 的命名空间
_NAMESPACE = uuid.UUID("6f1c9c2e-9b1a-4f3e-9a3a-2d6c1b0e7a55")


@dataclass
class IngestionResult:
    document_id: str
    chunks: int


class IngestionService:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vectorstore: VectorStore,
        splitter: RecursiveTextSplitter,
    ) -> None:
        self._embeddings = embeddings
        self._vectorstore = vectorstore
        self._splitter = splitter

    async def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        metadata: dict | None = None,
        tenant: str | None = None,
    ) -> IngestionResult:
        loader = get_loader(filename)
        sections = loader.load(data, filename)
        return await self._ingest_sections(sections, source=filename, metadata=metadata, tenant=tenant)

    async def ingest_text(
        self,
        text: str,
        *,
        source: str = "inline",
        metadata: dict | None = None,
        tenant: str | None = None,
    ) -> IngestionResult:
        sections = [LoadedSection(text=text)]
        return await self._ingest_sections(sections, source=source, metadata=metadata, tenant=tenant)

    async def _ingest_sections(
        self,
        sections: list[LoadedSection],
        *,
        source: str,
        metadata: dict | None,
        tenant: str | None,
    ) -> IngestionResult:
        # 1. 为本次入库分配唯一 document_id(用于后续按文档删除/溯源)
        document_id = uuid.uuid4().hex
        base_meta = dict(metadata or {})
        if tenant:
            base_meta["tenant"] = tenant

        # 2. 逐段分块,并为每个 chunk 组装元数据(来源、页码、序号、租户等)
        pieces: list[tuple[str, dict]] = []
        for section in sections:
            for chunk_text in self._splitter.split_text(section.text):
                meta = {
                    **base_meta,
                    **section.metadata,
                    "document_id": document_id,
                    "source": source,
                    "chunk_index": len(pieces),
                }
                pieces.append((chunk_text, meta))

        if not pieces:
            logger.warning("ingest_empty", extra={"extra_fields": {"source": source}})
            return IngestionResult(document_id=document_id, chunks=0)

        # 3. 批量向量化所有 chunk
        vectors = await self._embeddings.embed_documents([text for text, _ in pieces])
        # 4. 用 uuid5 生成稳定 point id(同一文档同一序号幂等),组装写入记录
        records = [
            VectorRecord(
                id=str(uuid.uuid5(_NAMESPACE, f"{document_id}:{index}")),
                vector=vector,
                text=text,
                metadata=meta,
            )
            for index, (vector, (text, meta)) in enumerate(zip(vectors, pieces))
        ]
        # 5. 写入向量库
        await self._vectorstore.upsert(records)
        logger.info(
            "ingest_done",
            extra={"extra_fields": {"document_id": document_id, "source": source, "chunks": len(records)}},
        )
        return IngestionResult(document_id=document_id, chunks=len(records))
