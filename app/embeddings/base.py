"""Embedding backend interface.

A small abstraction so the vector dimension/provider can change without touching
ingestion or retrieval. Implementations must embed documents and queries with the
*same* model so the vectors are comparable.
"""
from abc import ABC, abstractmethod


class Embedder(ABC):
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...
