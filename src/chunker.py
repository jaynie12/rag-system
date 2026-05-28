from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    word_count: int


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def chunk_text_by_paragraphs(text: str, max_words: int = 300) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph_words = _word_count(paragraph)

        # If paragraph alone exceeds the limit, flush current and keep the long paragraph as one chunk.
        if paragraph_words > max_words:
            if current_parts:
                joined = "\n\n".join(current_parts).strip()
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        text=joined,
                        word_count=_word_count(joined),
                    )
                )
                current_parts = []
                current_words = 0
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    text=paragraph,
                    word_count=paragraph_words,
                )
            )
            continue

        if current_parts and (current_words + paragraph_words) > max_words:
            joined = "\n\n".join(current_parts).strip()
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    text=joined,
                    word_count=_word_count(joined),
                )
            )
            current_parts = [paragraph]
            current_words = paragraph_words
        else:
            current_parts.append(paragraph)
            current_words += paragraph_words

    if current_parts:
        joined = "\n\n".join(current_parts).strip()
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                text=joined,
                word_count=_word_count(joined),
            )
        )

    return chunks
