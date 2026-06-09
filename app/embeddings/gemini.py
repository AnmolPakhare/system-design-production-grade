"""Gemini embeddings (gemini-embedding-001) with batching.

Subject to free-tier rate/daily limits; the ingestion pipeline paces batches.
Generation always uses Gemini regardless; this only governs retrieval vectors.
"""

from ..llm.gemini_client import GeminiClient
from .base import Embedder

_BATCH = 50


class GeminiEmbedder(Embedder):
    name = "gemini (gemini-embedding-001)"
    needs_pacing = True

    def __init__(self, client: GeminiClient, model: str) -> None:
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(
                self._client.embed(texts[i : i + _BATCH], self._model, "RETRIEVAL_DOCUMENT")
            )
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], self._model, "RETRIEVAL_QUERY")[0]
