from __future__ import annotations

from dataclasses import dataclass
import re

import tiktoken


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    word_count: int
    token_count: int


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_sentences(paragraph: str) -> list[str]:
    if not paragraph.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_chunk_units(paragraph: str) -> list[str]:
    """
    Split paragraph into chunk units while preserving quoted speech as atomic.

    Speech wrapped in quotes is kept together so chunking never breaks a speech
    block in half.
    """
    if not paragraph.strip():
        return []

    units: list[str] = []
    # Match double-quoted speech, including smart quotes.
    quote_pattern = re.compile(r'(".*?"|“.*?”)', re.DOTALL)
    cursor = 0

    for match in quote_pattern.finditer(paragraph):
        before = paragraph[cursor : match.start()].strip()
        if before:
            units.extend(_split_sentences(before))

        speech = match.group(0).strip()
        if speech:
            units.append(speech)
        cursor = match.end()

    tail = paragraph[cursor:].strip()
    if tail:
        units.extend(_split_sentences(tail))
    return units


def _is_quoted_speech(text: str) -> bool:
    stripped = text.strip()
    return (
        (stripped.startswith('"') and stripped.endswith('"'))
        or (stripped.startswith("“") and stripped.endswith("”"))
    )


def _build_chunk(text: str, chunk_index: int, encoding: tiktoken.Encoding) -> Chunk:
    cleaned = text.strip()
    return Chunk(
        chunk_index=chunk_index,
        text=cleaned,
        word_count=_word_count(cleaned),
        token_count=len(encoding.encode(cleaned)),
    )


def chunk_text_by_sentences(text: str, max_tokens: int = 800) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    paragraphs = _split_paragraphs(text)
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[Chunk] = []
    current_parts: list[str] = []

    for paragraph in paragraphs:
        units = _split_chunk_units(paragraph)
        for unit in units:
            unit_tokens = encoding.encode(unit)
            unit_token_count = len(unit_tokens)

            if unit_token_count > max_tokens and not _is_quoted_speech(unit):
                if current_parts:
                    chunks.append(_build_chunk(" ".join(current_parts), len(chunks), encoding))
                    current_parts = []

                for start in range(0, unit_token_count, max_tokens):
                    token_slice = unit_tokens[start : start + max_tokens]
                    slice_text = encoding.decode(token_slice).strip()
                    if not slice_text:
                        continue
                    chunks.append(_build_chunk(slice_text, len(chunks), encoding))
                continue

            if current_parts:
                projected_text = " ".join([*current_parts, unit])
                projected_tokens = len(encoding.encode(projected_text))
            else:
                projected_tokens = unit_token_count

            if current_parts and projected_tokens > max_tokens:
                chunks.append(_build_chunk(" ".join(current_parts), len(chunks), encoding))
                current_parts = [unit]
                continue

            current_parts.append(unit)

    if current_parts:
        chunks.append(_build_chunk(" ".join(current_parts), len(chunks), encoding))

    return chunks
