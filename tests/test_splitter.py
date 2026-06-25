from app.ingestion.splitter import RecursiveTextSplitter


def test_empty_text_returns_empty():
    assert RecursiveTextSplitter(100, 20).split_text("") == []
    assert RecursiveTextSplitter(100, 20).split_text("   ") == []


def test_short_text_single_chunk():
    chunks = RecursiveTextSplitter(100, 20).split_text("这是一段很短的文本。")
    assert chunks == ["这是一段很短的文本。"]


def test_long_text_is_split_into_multiple_chunks():
    text = "。".join(f"第{i}句话内容" for i in range(50)) + "。"
    chunks = RecursiveTextSplitter(60, 10).split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 60 + 10 for c in chunks)  # 允许重叠带来的少量超出


def test_oversized_segment_is_hard_wrapped():
    text = "x" * 250  # 无分隔符的超长串
    chunks = RecursiveTextSplitter(100, 0).split_text(text)
    assert len(chunks) == 3
    assert "".join(chunks) == text


def test_overlap_carries_context():
    text = "段落一内容。" * 20
    splitter = RecursiveTextSplitter(50, 15)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    # 第二块开头应包含来自第一块结尾的重叠内容
    assert chunks[1][:5] in chunks[0]


def test_overlap_must_be_smaller_than_size():
    try:
        RecursiveTextSplitter(50, 50)
        assert False, "应抛出异常"
    except ValueError:
        pass
