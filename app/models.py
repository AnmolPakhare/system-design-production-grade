"""Pydantic request/response schemas for the API (the public contract)."""

from pydantic import BaseModel, Field


class Source(BaseModel):
    source: str
    path: str
    heading: str = ""


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    # Prior turns as (user, assistant) pairs for multi-turn memory.
    history: list[tuple[str, str]] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    search_query: str
    model: str
    request_id: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    vectorstore: str
    embedding_provider: str
    collection_count: int
    generator_model: str
