import json

import httpx
import pytest

from app.core.embeddings.tei import TEIEmbedding
from app.exceptions import RateLimitedError, UpstreamError


def _make_embedding(handler, *, dim=4, batch_size=2):
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://tei", transport=transport)
    return TEIEmbedding(base_url="http://tei", dim=dim, batch_size=batch_size, client=client)


async def test_embed_documents_batches_and_preserves_order():
    seen_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        inputs = payload["inputs"]
        seen_batches.append(inputs)
        # 每个输入返回一个可区分的向量
        return httpx.Response(200, json=[[float(len(t))] * 4 for t in inputs])

    emb = _make_embedding(handler, dim=4, batch_size=2)
    vectors = await emb.embed_documents(["a", "bb", "ccc"])

    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)
    # batch_size=2 -> 两批:["a","bb"] 与 ["ccc"]
    assert seen_batches == [["a", "bb"], ["ccc"]]
    # 顺序与入参一致(向量首元素 = 文本长度)
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


async def test_embed_query():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.5, 0.5, 0.5, 0.5]])

    emb = _make_embedding(handler)
    vec = await emb.embed_query("你好")
    assert vec == [0.5, 0.5, 0.5, 0.5]


async def test_rate_limit_maps_to_framework_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many requests"})

    emb = _make_embedding(handler)
    with pytest.raises(RateLimitedError):
        await emb.embed_query("x")


async def test_server_error_maps_to_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    emb = _make_embedding(handler)
    with pytest.raises(UpstreamError):
        await emb.embed_query("x")


async def test_empty_input_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:  # 不应被调用
        raise AssertionError("不应发起请求")

    emb = _make_embedding(handler)
    assert await emb.embed_documents([]) == []
