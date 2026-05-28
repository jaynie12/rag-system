from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_embedding_model: str
    openai_chat_model: str
    chunk_max_tokens: int
    retrieval_top_k: int
    min_retrieval_score: float
    db_path: Path
    data_dir: Path


def _require_str(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got: {value}")
    return value


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got: {raw}") from exc
    return value


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        openai_api_key=_require_str("OPENAI_API_KEY"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        chunk_max_tokens=_parse_int("CHUNK_MAX_TOKENS", 800),
        retrieval_top_k=_parse_int("RETRIEVAL_TOP_K", 5),
        min_retrieval_score=_parse_float("MIN_RETRIEVAL_SCORE", 0.2),
        db_path=Path(os.getenv("DB_PATH", "storage/rag.db")).expanduser(),
        data_dir=Path(os.getenv("DATA_DIR", "data")).expanduser(),
    )

    if settings.retrieval_top_k < 1:
        raise ValueError("RETRIEVAL_TOP_K must be >= 1")
    if settings.chunk_max_tokens < 1:
        raise ValueError("CHUNK_MAX_TOKENS must be >= 1")
    if settings.min_retrieval_score < -1.0 or settings.min_retrieval_score > 1.0:
        raise ValueError("MIN_RETRIEVAL_SCORE must be between -1 and 1")

    return settings
