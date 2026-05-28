from __future__ import annotations

from pathlib import Path

from src.chunker import chunk_text_by_sentences
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.generator import AnswerGenerator
from src.loader import load_document_text
from src.retriever import RetrievedChunk, Retriever
from src.vector_store import SQLiteVectorStore


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = SQLiteVectorStore(settings.db_path)
        self.embedding_client = EmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
        self.retriever = Retriever(self.store, self.embedding_client)
        self.generator = AnswerGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
            min_retrieval_score=settings.min_retrieval_score,
        )

    def build_index(self, pdf_path: Path, title: str | None = None) -> int:
        text = load_document_text(pdf_path)
        chunks = chunk_text_by_sentences(text, max_tokens=self.settings.chunk_max_tokens)
        vectors = self.embedding_client.embed_texts([chunk.text for chunk in chunks])

        document_title = title or pdf_path.stem
        doc_id = self.store.upsert_document(document_title, str(pdf_path))
        self.store.replace_document_chunks(doc_id, chunks, vectors)
        return len(chunks)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, top_k=self.settings.retrieval_top_k)

    def answer_question(self, question: str) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.retrieve(question)
        answer = self.generator.answer(question, retrieved)
        return answer, retrieved
