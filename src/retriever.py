from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embeddings import EmbeddingClient
from src.vector_store import SQLiteVectorStore, StoredChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float
    chunk_index: int
    document_id: int


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Retriever:
    def __init__(self, store: SQLiteVectorStore, embedding_client: EmbeddingClient) -> None:
        self._store = store
        self._embedding_client = embedding_client

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        candidates = self._store.fetch_all_chunks_with_vectors()
        if not candidates:
            return []
        query_vector = self._embedding_client.embed_text(query)
        scored = [
            self._score_candidate(candidate=c, query_vector=query_vector)
            for c in candidates
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _score_candidate(candidate: StoredChunk, query_vector: list[float]) -> RetrievedChunk:
        score = cosine_similarity(query_vector, candidate.vector)
        return RetrievedChunk(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=score,
            chunk_index=candidate.chunk_index,
            document_id=candidate.document_id,
        )
