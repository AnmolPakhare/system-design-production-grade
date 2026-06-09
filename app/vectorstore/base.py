"""Vector store interface.

Abstracting the store lets us swap the local Chroma file for a managed/served
backend (Qdrant, pgvector, Pinecone, Vertex) without touching ingest/retrieval.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Hit:
    document: str
    metadata: dict
    embedding: list[float]


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids, embeddings, documents, metadatas) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], n_results: int) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def existing_ids(self) -> set: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...
