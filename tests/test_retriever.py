from pathlib import Path

from src.retriever import Retriever, cosine_similarity
from src.vector_store import SQLiteVectorStore
from src.chunker import Chunk


class FakeEmbeddingClient:
    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = query_vector

    def embed_text(self, _: str) -> list[float]:
        return self._query_vector


def test_cosine_similarity_orders_expected_scores() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) > cosine_similarity([1.0, 0.0], [0.0, 1.0])


def test_retriever_returns_top_ranked_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = SQLiteVectorStore(db_path)
    doc_id = store.upsert_document("Pride and Prejudice", "data/book.pdf")

    chunks = [
        Chunk(chunk_index=0, text="Elizabeth Bennet appears.", word_count=3, token_count=5),
        Chunk(chunk_index=1, text="Mr Darcy appears.", word_count=3, token_count=4),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store.replace_document_chunks(doc_id, chunks, vectors)

    retriever = Retriever(store, FakeEmbeddingClient([1.0, 0.0]))  # type: ignore[arg-type]
    results = retriever.retrieve("Who is Elizabeth?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_index == 0
    assert "Elizabeth" in results[0].text


def test_retriever_returns_empty_when_no_index(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "empty.db")
    retriever = Retriever(store, FakeEmbeddingClient([1.0, 0.0]))  # type: ignore[arg-type]
    results = retriever.retrieve("Any question", top_k=3)
    assert results == []
