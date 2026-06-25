from fastapi.testclient import TestClient

from app.config import Settings
from tests.conftest import FakeVectorStore


def test_healthz(app_factory, base_settings):
    app = app_factory(base_settings)
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_readyz_ready_and_unready(app_factory, base_settings):
    app = app_factory(base_settings, vectorstore=FakeVectorStore(healthy=True))
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200

    app2 = app_factory(base_settings, vectorstore=FakeVectorStore(healthy=False))
    with TestClient(app2) as client:
        assert client.get("/readyz").status_code == 503


def test_chat_returns_answer_and_citations(app_factory, base_settings):
    app = app_factory(base_settings)
    with TestClient(app) as client:
        resp = client.post("/v1/chat", json={"query": "付款期限?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["conversation_id"]
        assert body["citations"][0]["source"] == "合同.pdf"


def test_chat_validation_error(app_factory, base_settings):
    app = app_factory(base_settings)
    with TestClient(app) as client:
        resp = client.post("/v1/chat", json={"query": ""})
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_request"


def test_ingest_text(app_factory, base_settings):
    app = app_factory(base_settings)
    with TestClient(app) as client:
        resp = client.post("/v1/documents/text", json={"text": "一些知识内容"})
        assert resp.status_code == 200
        assert resp.json() == {"document_id": "doc-text", "chunks": 1}


def test_conversation_lifecycle(app_factory, base_settings):
    app = app_factory(base_settings)
    with TestClient(app) as client:
        chat = client.post("/v1/chat", json={"query": "问题一"})
        cid = chat.json()["conversation_id"]
        got = client.get(f"/v1/conversations/{cid}")
        assert got.status_code == 200
        assert len(got.json()["messages"]) == 2  # user + assistant
        assert client.delete(f"/v1/conversations/{cid}").json() == {"status": "cleared"}
        assert client.get(f"/v1/conversations/{cid}").json()["messages"] == []


def test_auth_required_when_keys_configured(app_factory):
    settings = Settings(api_keys=["secret-key"], memory_backend="memory")
    app = app_factory(settings)
    with TestClient(app) as client:
        # 缺少 Key
        assert client.post("/v1/chat", json={"query": "x"}).status_code == 401
        # 错误 Key
        assert client.post(
            "/v1/chat", json={"query": "x"}, headers={"X-API-Key": "wrong"}
        ).status_code == 401
        # 正确 Key
        ok = client.post("/v1/chat", json={"query": "x"}, headers={"X-API-Key": "secret-key"})
        assert ok.status_code == 200
