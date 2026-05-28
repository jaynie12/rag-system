from __future__ import annotations

from openai import OpenAI

from src.retriever import RetrievedChunk

FALLBACK_ANSWER = "I don't know"


class AnswerGenerator:
    def __init__(self, api_key: str, model: str, min_retrieval_score: float = 0.2) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._min_retrieval_score = min_retrieval_score

    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return FALLBACK_ANSWER

        best_score = max(chunk.score for chunk in retrieved_chunks)
        if best_score < self._min_retrieval_score:
            return FALLBACK_ANSWER

        context = "\n\n".join(
            f"[chunk {chunk.chunk_index} | score {chunk.score:.3f}]\n{chunk.text}"
            for chunk in retrieved_chunks
        )

        system_prompt = (
            "You are a book QA assistant. Answer only with evidence from provided context. "
            "If the answer is not clearly present in the context, reply exactly: I don't know. "
            "Do not use outside knowledge."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip() if response.choices else ""
        if not answer:
            return FALLBACK_ANSWER
        return answer
