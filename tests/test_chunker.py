import tiktoken

from src.chunker import chunk_text_by_sentences


def _token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def test_chunker_groups_sentences_until_token_limit() -> None:
    first = "Elizabeth admired the countryside."
    second = "Jane remained calm and thoughtful."
    third = "Mr. Darcy said very little at dinner."
    text = f"{first} {second} {third}"
    first_two = f"{first} {second}"
    max_tokens = _token_count(first_two)

    chunks = chunk_text_by_sentences(text, max_tokens=max_tokens)

    assert len(chunks) == 2
    assert chunks[0].text == first_two
    assert chunks[1].text == third
    assert chunks[0].token_count <= max_tokens
    assert chunks[1].token_count <= max_tokens


def test_chunker_starts_new_chunk_at_sentence_boundary() -> None:
    first = "The rain ended quickly."
    second = "Everyone returned to the ballroom."
    third = "The music resumed."
    text = f"{first} {second} {third}"
    max_tokens = _token_count(f"{first} {second}")

    chunks = chunk_text_by_sentences(text, max_tokens=max_tokens)

    assert chunks[0].text.endswith(".")
    assert chunks[1].text == third


def test_chunker_splits_single_overlong_sentence_by_tokens() -> None:
    long_sentence = ("token " * 1200).strip() + "."
    max_tokens = 100

    chunks = chunk_text_by_sentences(long_sentence, max_tokens=max_tokens)

    assert len(chunks) > 1
    assert all(chunk.token_count <= max_tokens for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunker_does_not_split_quoted_speech() -> None:
    speech = '"' + ("I insist we keep speaking without interruption. " * 120).strip() + '"'
    text = f"Before the speech. {speech} After the speech."
    max_tokens = 120

    chunks = chunk_text_by_sentences(text, max_tokens=max_tokens)

    speech_chunks = [chunk for chunk in chunks if speech in chunk.text]
    assert len(speech_chunks) == 1
    assert speech_chunks[0].text == speech
