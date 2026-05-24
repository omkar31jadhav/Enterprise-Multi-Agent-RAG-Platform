import pytest

from text_processor import split_into_chunks


def test_split_into_chunks_preserves_overlap() -> None:
    text = "one two three four five"

    chunks = split_into_chunks(text, chunk_size=3, overlap=1)

    assert chunks == ["one two three", "three four five", "five"]


def test_split_into_chunks_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap must be smaller"):
        split_into_chunks("one two three", chunk_size=2, overlap=2)
