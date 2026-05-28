from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.chunker import Chunk


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    text: str
    word_count: int
    vector: list[float]


class SQLiteVectorStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL UNIQUE,
                    source_path TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    UNIQUE(document_id, chunk_index),
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def upsert_document(self, title: str, source_path: str) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (title, source_path)
                VALUES (?, ?)
                ON CONFLICT(title) DO UPDATE SET source_path=excluded.source_path
                """,
                (title, source_path),
            )
            doc_id = conn.execute("SELECT id FROM documents WHERE title = ?", (title,)).fetchone()[0]
            conn.commit()
            return int(doc_id)

    def replace_document_chunks(
        self,
        document_id: int,
        chunks: Iterable[Chunk],
        vectors: Iterable[list[float]],
    ) -> None:
        chunk_list = list(chunks)
        vector_list = list(vectors)
        if len(chunk_list) != len(vector_list):
            raise ValueError("chunks and vectors must have the same length")

        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
            existing_ids = [row[0] for row in existing]
            if existing_ids:
                placeholders = ",".join(["?"] * len(existing_ids))
                conn.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", existing_ids)
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

            for chunk, vector in zip(chunk_list, vector_list):
                cursor = conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, word_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, chunk.chunk_index, chunk.text, chunk.word_count),
                )
                chunk_id = cursor.lastrowid
                conn.execute(
                    """
                    INSERT INTO embeddings (chunk_id, vector_json)
                    VALUES (?, ?)
                    """,
                    (chunk_id, json.dumps(vector)),
                )
            conn.commit()

    def list_documents(self) -> list[tuple[int, str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, title, source_path FROM documents ORDER BY id").fetchall()
        return [(int(row[0]), str(row[1]), str(row[2])) for row in rows]

    def fetch_all_chunks_with_vectors(self) -> list[StoredChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.text, c.word_count, e.vector_json
                FROM chunks c
                JOIN embeddings e ON e.chunk_id = c.id
                ORDER BY c.document_id, c.chunk_index
                """
            ).fetchall()
        result: list[StoredChunk] = []
        for row in rows:
            result.append(
                StoredChunk(
                    chunk_id=int(row[0]),
                    document_id=int(row[1]),
                    chunk_index=int(row[2]),
                    text=str(row[3]),
                    word_count=int(row[4]),
                    vector=[float(v) for v in json.loads(row[5])],
                )
            )
        return result
