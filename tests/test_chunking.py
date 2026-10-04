import pytest

from evidencedesk.chunking import chunk_text


def test_whitespace_only_text_yields_no_chunks() -> None:
    assert chunk_text("   \n\t  ", size=100, overlap=10) == []


def test_chunks_cover_all_words_and_are_numbered() -> None:
    words = [f"wort{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), size=120, overlap=20)

    assert len(chunks) > 1
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    combined = " ".join(chunk.content for chunk in chunks)
    for word in words:
        assert word in combined


def test_words_are_never_split() -> None:
    chunks = chunk_text(" ".join(["alpha"] * 50), size=40, overlap=8)
    for chunk in chunks:
        assert "alpha" in chunk.content or len(chunk.content) >= len("alpha")


def test_overlap_near_size_still_terminates() -> None:
    chunks = chunk_text(" ".join(f"w{i}" for i in range(50)), size=10, overlap=9)
    assert chunks
    assert chunks[-1].content


def test_single_word_longer_than_size_is_kept() -> None:
    chunks = chunk_text("x" * 500, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].content == "x" * 500


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=-1)
