import pytest

from app.config import Settings
from app.memory.memory_store import InMemoryConversationStore
from app.services.rag_service import RAGService
from tests.conftest import FakeLLM, FakeRetriever


@pytest.fixture
def settings():
    return Settings(memory_backend="memory", max_history_turns=5)


@pytest.fixture
def service(settings):
    return RAGService(FakeLLM("付款期限为 30 日 [1]"), FakeRetriever(), InMemoryConversationStore(), settings)


async def test_answer_returns_citations(service):
    result = await service.answer("付款期限?")
    assert result.answer == "付款期限为 30 日 [1]"
    assert result.conversation_id
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.index == 1
    assert citation.source == "合同.pdf"
    assert citation.page == 2


async def test_answer_persists_history(service):
    first = await service.answer("第一问")
    cid = first.conversation_id
    # 复用同一会话
    await service.answer("第二问", conversation_id=cid)
    history = await service._memory.load(cid)
    # 两轮各 user+assistant 共 4 条
    assert len(history) == 4
    assert history[0].role == "user" and history[0].content == "第一问"


async def test_answer_stream_emits_events(service):
    events = [event async for event in service.answer_stream("付款期限?")]
    types = [e["type"] for e in events]
    assert types[0] == "token"
    assert "citations" in types
    assert types[-1] == "done"
    tokens = "".join(e["data"] for e in events if e["type"] == "token")
    assert tokens  # 有内容产出
    citations_event = next(e for e in events if e["type"] == "citations")
    assert citations_event["data"][0]["source"] == "合同.pdf"
