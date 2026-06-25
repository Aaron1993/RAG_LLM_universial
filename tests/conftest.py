"""测试夹具:以 fakes 替换外部依赖(LLM / 检索 / 向量库),不触网。"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from app.api import deps
from app.config import Settings
from app.core.vectorstore.base import RetrievedChunk
from app.ingestion.service import IngestionResult
from app.main import create_app
from app.memory.memory_store import InMemoryConversationStore
from app.services.rag_service import RAGService


class FakeLLM:
    def __init__(self, reply: str = "这是回答 [1]") -> None:
        self.reply = reply
        self.last_messages = None

    async def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        self.last_messages = list(messages)
        return self.reply

    async def chat_stream(self, messages, *, temperature=None, max_tokens=None) -> AsyncIterator[str]:
        self.last_messages = list(messages)
        for piece in ["这是", "回答", " [1]"]:
            yield piece


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [
            RetrievedChunk(
                id="p1",
                text="合同自签署之日起生效。",
                score=0.91,
                metadata={"document_id": "d1", "source": "合同.pdf", "page": 2},
            )
        ]
        self.last_query = None

    async def retrieve(self, query, *, top_k=None, filters=None) -> list[RetrievedChunk]:
        self.last_query = query
        return self.chunks


class FakeVectorStore:
    def __init__(self, healthy: bool = True) -> None:
        self._healthy = healthy
        self.deleted: list = []

    async def ensure_collection(self, dim: int) -> None:
        return None

    async def upsert(self, records) -> None:
        return None

    async def search(self, vector, *, top_k, filters=None, score_threshold=0.0):
        return []

    async def delete(self, *, ids=None, filters=None) -> None:
        self.deleted.append((ids, filters))

    async def health(self) -> bool:
        return self._healthy

    async def close(self) -> None:
        return None


class FakeIngestionService:
    async def ingest_bytes(self, data, filename, *, metadata=None, tenant=None) -> IngestionResult:
        return IngestionResult(document_id="doc-file", chunks=3)

    async def ingest_text(self, text, *, source="inline", metadata=None, tenant=None) -> IngestionResult:
        return IngestionResult(document_id="doc-text", chunks=1)


@pytest.fixture
def base_settings() -> Settings:
    return Settings(api_keys=[], memory_backend="memory", environment="dev", dashscope_api_key="test")


def _build_app(settings: Settings, *, vectorstore: FakeVectorStore | None = None):
    app = create_app(settings)
    memory = InMemoryConversationStore()
    fake_llm = FakeLLM()
    rag = RAGService(fake_llm, FakeRetriever(), memory, settings)
    app.state.rag_service = rag
    app.state.memory = memory
    app.state.vectorstore = vectorstore or FakeVectorStore()
    app.state.ingestion_service = FakeIngestionService()
    app.state.embeddings = None
    app.state.llm = fake_llm
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    return app


@pytest.fixture
def app_factory():
    """返回一个可定制 settings / vectorstore 的 app 构造器。"""
    return _build_app
