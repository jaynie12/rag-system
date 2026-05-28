from src.chunker import chunk_text_by_paragraphs


def _words(count: int, token: str) -> str:
    return " ".join([token] * count)


def test_chunker_groups_paragraphs_until_limit() -> None:
    text = f"{_words(120, 'a')}\n\n{_words(100, 'b')}\n\n{_words(50, 'c')}"
    chunks = chunk_text_by_paragraphs(text, max_words=300)

    assert len(chunks) == 1
    assert chunks[0].word_count == 270


def test_chunker_starts_new_chunk_when_next_paragraph_overflows() -> None:
    text = f"{_words(180, 'a')}\n\n{_words(140, 'b')}\n\n{_words(60, 'c')}"
    chunks = chunk_text_by_paragraphs(text, max_words=300)

    assert len(chunks) == 2
    assert chunks[0].word_count == 180
    assert chunks[1].word_count == 200


def test_chunker_keeps_very_long_paragraph_as_single_chunk() -> None:
    text = f"{_words(320, 'a')}\n\n{_words(50, 'b')}"
    chunks = chunk_text_by_paragraphs(text, max_words=300)

    assert len(chunks) == 2
    assert chunks[0].word_count == 320
    assert chunks[1].word_count == 50
