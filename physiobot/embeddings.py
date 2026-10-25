"""Gemini embeddings, using gemini-embedding-2's asymmetric prompt format."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

from google import genai
from google.genai import errors, types

from .config import get_settings

BATCH_SIZE = 100


def query_text(question: str) -> str:
    return f"task: question answering | query: {question}"


def document_text(content: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {content}"


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    def __init__(self, client: genai.Client | None = None):
        settings = get_settings()
        self.client = client or genai.Client(api_key=settings.google_api_key)
        self.model = settings.embedding_model
        self.config = types.EmbedContentConfig(output_dimensionality=settings.embedding_dim)

    def _embed(self, texts: list[str], retries: int = 5) -> list[list[float]]:
        for attempt in range(retries + 1):
            try:
                response = self.client.models.embed_content(model=self.model, contents=texts, config=self.config)
                break
            except errors.APIError as e:
                # Free-tier quotas are per minute: back off and retry on 429/503.
                if e.code not in (429, 503) or attempt == retries:
                    raise
                time.sleep(min(60, 2 ** (attempt + 2)))
        vectors = [e.values for e in (response.embeddings or [])]
        if len(vectors) != len(texts) or any(v is None for v in vectors):
            raise RuntimeError("Embedding API returned an unexpected number of vectors")
        return vectors  # type: ignore[return-value]

    def embed_query(self, question: str) -> list[float]:
        return self._embed([query_text(question)])[0]

    def embed_documents(self, chunks: Iterable[str], title: str | None = None) -> list[list[float]]:
        texts = [document_text(c, title) for c in chunks]
        vectors: list[list[float]] = []
        for batch in _batches(texts, BATCH_SIZE):
            vectors.extend(self._embed(batch))
        return vectors
