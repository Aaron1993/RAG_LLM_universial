import json

import pytest

from app.core.vectorstore.base import VectorRecord
from app.core.vectorstore.redis_store import RedisStackVectorStore


class FakePipeline:
    def __init__(self, parent: "FakeRedis") -> None:
        self._parent = parent
        self._ops: list[tuple] = []

    def execute_command(self, *args):
        self._ops.append(args)
        return self

    async def execute(self):
        for args in self._ops:
            await self._parent.execute_command(*args)


class FakeRedis:
    def __init__(self, *, knn_response=None, nocontent_response=None, index_exists=False):
        self.calls: list[tuple] = []
        self.json_store: dict[str, str] = {}
        self.deleted: list[str] = []
        self._knn = knn_response
        self._nocontent = nocontent_response
        self._index_exists = index_exists

    async def execute_command(self, *args):
        self.calls.append(args)
        cmd = args[0]
        if cmd == "FT.INFO":
            if self._index_exists:
                return ["index_name", args[1]]
            raise RuntimeError("Unknown index name")
        if cmd == "FT.CREATE":
            self._index_exists = True
            return "OK"
        if cmd == "JSON.SET":
            self.json_store[args[1]] = args[3]
            return "OK"
        if cmd == "FT.SEARCH":
            return self._nocontent if "NOCONTENT" in args else self._knn
        if cmd == "JSON.GET":
            return self.json_store.get(args[1])
        return None

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self)

    async def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    async def ping(self):
        return True

    async def aclose(self):
        return None


def _store(client: FakeRedis) -> RedisStackVectorStore:
    return RedisStackVectorStore(index_name="idx", prefix="embedding:", client=client)


async def test_ensure_collection_creates_index_with_dim():
    client = FakeRedis(index_exists=False)
    await _store(client).ensure_collection(1024)
    create = next(c for c in client.calls if c[0] == "FT.CREATE")
    assert "DIM" in create and "1024" in create
    assert "COSINE" in create
    # 默认 TAG 字段都建了索引
    assert "document_id" in create and "source" in create and "tenant" in create


async def test_ensure_collection_skips_when_exists():
    client = FakeRedis(index_exists=True)
    await _store(client).ensure_collection(1024)
    assert not any(c[0] == "FT.CREATE" for c in client.calls)


async def test_upsert_writes_json_with_vector_and_metadata():
    client = FakeRedis()
    store = _store(client)
    await store.upsert(
        [VectorRecord(id="p1", vector=[0.1, 0.2], text="内容", metadata={"source": "a.pdf", "page": 2})]
    )
    assert "embedding:p1" in client.json_store
    doc = json.loads(client.json_store["embedding:p1"])
    assert doc["text"] == "内容"
    assert doc["vector"] == [0.1, 0.2]
    assert doc["metadata"]["source"] == "a.pdf"


async def test_search_parses_distance_to_similarity():
    doc = json.dumps({"text": "命中内容", "vector": [0.0], "metadata": {"source": "a.pdf", "page": 3}})
    # vector_score(余弦距离)=0.1 -> 相似度 (2-0.1)/2 = 0.95
    knn = [1, "embedding:p1", ["vector_score", "0.1", "$", doc]]
    client = FakeRedis(knn_response=knn)
    chunks = await _store(client).search([0.1, 0.2], top_k=3)
    assert len(chunks) == 1
    assert chunks[0].text == "命中内容"
    assert chunks[0].metadata["source"] == "a.pdf"
    assert abs(chunks[0].score - 0.95) < 1e-6


async def test_search_applies_filter_and_threshold():
    knn = [1, "embedding:p1", ["vector_score", "1.5", "$", json.dumps({"text": "x", "metadata": {}})]]
    client = FakeRedis(knn_response=knn)
    # 相似度 (2-1.5)/2 = 0.25 < 阈值 0.5 -> 被过滤
    chunks = await _store(client).search([0.1], top_k=3, filters={"tenant": "acme"}, score_threshold=0.5)
    assert chunks == []
    # 校验过滤表达式进入了 KNN 查询串
    search_call = next(c for c in client.calls if c[0] == "FT.SEARCH")
    knn_query = search_call[2]
    assert "@tenant:{acme}" in knn_query
    assert "=>[KNN 3 @vector $vec" in knn_query


async def test_delete_by_ids():
    client = FakeRedis()
    await _store(client).delete(ids=["p1", "p2"])
    assert client.deleted == ["embedding:p1", "embedding:p2"]


async def test_delete_by_filters_searches_then_deletes():
    client = FakeRedis(nocontent_response=[1, "embedding:p9"])
    await _store(client).delete(filters={"document_id": "d1"})
    assert client.deleted == ["embedding:p9"]
