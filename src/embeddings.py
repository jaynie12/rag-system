from __future__ import annotations

import time
from typing import Iterable

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed_texts(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        max_retries: int = 3,
    ) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors.extend(self._embed_with_retry(batch, max_retries=max_retries))
        return vectors

    def embed_text(self, text: str, max_retries: int = 3) -> list[float]:
        return self._embed_with_retry([text], max_retries=max_retries)[0]

    def _embed_with_retry(self, texts: list[str], max_retries: int) -> list[list[float]]:
        delay = 1.0
        for attempt in range(max_retries + 1):
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except Exception:
                if attempt >= max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("Embedding retry loop failed unexpectedly")
