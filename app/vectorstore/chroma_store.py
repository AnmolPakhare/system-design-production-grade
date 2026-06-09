"""Chroma-backed vector store (persistent, cosine space)."""

from ..config import Settings
from .base import Hit, VectorStore


class ChromaStore(VectorStore):
    def __init__(self, settings: Settings) -> None:
        import chromadb

        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self._col = self._client.get_or_create_collection(
            name=settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        self._col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], n_results: int) -> list[Hit]:
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "embeddings"],
        )
        if not res["documents"] or not res["documents"][0]:
            return []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        embs = res["embeddings"][0]
        return [Hit(d, m, list(e)) for d, m, e in zip(docs, metas, embs, strict=False)]

    def count(self) -> int:
        return self._col.count()

    def existing_ids(self) -> set:
        return set(self._col.get(include=[])["ids"])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=ids)


def get_vectorstore(settings: Settings) -> VectorStore:
    if settings.vectorstore == "chroma":
        return ChromaStore(settings)
    raise ValueError(f"Unknown vectorstore: {settings.vectorstore}")
