"""Composition root: build the RAG pipeline and its dependencies once.

The LLM client is created lazily on first generation so the service (and tests)
can start, serve /health, and run without a Gemini key until a real query needs
generation.
"""
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..embeddings import get_embedder
from ..rag.pipeline import RagPipeline
from ..rag.retriever import Retriever
from ..vectorstore import VectorStore, get_vectorstore


@dataclass
class Container:
    settings: Settings
    store: VectorStore
    pipeline: RagPipeline


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    embedder = get_embedder(settings)
    store = get_vectorstore(settings)
    retriever = Retriever(store, embedder, settings)

    _client = {}

    def generate(prompt: str, model: str | None = None, max_retries: int = 6) -> str:
        if "c" not in _client:
            from ..llm.gemini_client import GeminiClient

            _client["c"] = GeminiClient(settings)
        return _client["c"].generate(prompt, model, max_retries)

    pipeline = RagPipeline(retriever, generate, settings)
    return Container(settings=settings, store=store, pipeline=pipeline)
