from app.core.vectorstore.base import RetrievedChunk
from app.memory.base import StoredMessage
from app.prompts.templates import build_context_block, build_messages


def _chunks():
    return [
        RetrievedChunk(id="1", text="甲方应在 30 日内付款。", score=0.9, metadata={"source": "合同.pdf", "page": 3}),
        RetrievedChunk(id="2", text="违约金为合同金额的 5%。", score=0.8, metadata={"source": "合同.pdf"}),
    ]


def test_context_block_is_numbered():
    block = build_context_block(_chunks())
    assert "[1]" in block and "[2]" in block
    assert "第 3 页" in block
    assert "甲方应在 30 日内付款。" in block


def test_build_messages_structure():
    history = [StoredMessage("user", "你好"), StoredMessage("assistant", "您好")]
    messages = build_messages("付款期限是多久?", _chunks(), history)
    assert messages[0]["role"] == "system"
    # 历史被保留
    assert messages[1] == {"role": "user", "content": "你好"}
    assert messages[2] == {"role": "assistant", "content": "您好"}
    # 最后一条是带上下文的用户问题
    assert messages[-1]["role"] == "user"
    assert "付款期限是多久?" in messages[-1]["content"]
    assert "[1]" in messages[-1]["content"]


def test_build_messages_without_chunks():
    messages = build_messages("无关问题", [], [])
    assert "未检索到相关资料" in messages[-1]["content"]
