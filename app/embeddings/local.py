"""Offline local embeddings via Chroma's bundled all-MiniLM-L6-v2 (ONNX).

No API key, no quota, deterministic — the production default so ingestion and
serving never block on external rate limits.
"""

from .base import Embedder


class LocalEmbedder(Embedder):
    name = "local (all-MiniLM-L6-v2)"

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._ef([text])[0]))
