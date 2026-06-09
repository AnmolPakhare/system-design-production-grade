"""Embedder factory."""
from ..config import Settings
from .base import Embedder


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "gemini":
        from ..llm.gemini_client import GeminiClient
        from .gemini import GeminiEmbedder

        return GeminiEmbedder(GeminiClient(settings), settings.embedding_model)
    from .local import LocalEmbedder

    return LocalEmbedder()


__all__ = ["Embedder", "get_embedder"]
