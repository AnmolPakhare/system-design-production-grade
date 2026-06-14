"""Typed, environment-driven configuration (12-factor).

All settings are read from environment variables prefixed with ``RAG_`` (or a
local ``.env`` file). Centralising config here keeps secrets out of code and
makes the service configurable per-environment without code changes.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        # Absolute path so the .env is found no matter which directory the
        # server/CLI is launched from (pydantic resolves a bare ".env" against
        # the process CWD, which breaks `ragctl serve` started elsewhere).
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    judge_model: str = "gemini-2.5-flash"

    # --- Embeddings ---
    embedding_provider: str = "local"  # "local" | "gemini"
    embedding_model: str = "gemini-embedding-001"

    # --- Vector store ---
    vectorstore: str = "chroma"
    chroma_dir: str = "data/chroma"
    collection: str = "system_design"

    # --- Retrieval ---
    top_k: int = 6
    candidate_multiplier: int = 5
    dup_cosine_threshold: float = 0.92
    max_per_source: int = 2

    # --- Chunking ---
    chunk_size: int = 4000
    chunk_overlap: int = 300

    # --- Service ---
    api_key: str = ""  # if set, X-API-Key header is required
    log_level: str = "INFO"
    enable_guardrails: bool = True

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()
