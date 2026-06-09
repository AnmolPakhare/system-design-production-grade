"""RAG orchestration: condense -> retrieve -> guardrail -> generate -> flag.

The LLM is injected as a ``generate`` callable so the pipeline can be unit-tested
without any network/quota (a stub generator).
"""
from collections.abc import Callable

from ..config import Settings
from .guardrails import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    Turn,
    condense_query,
    has_sufficient_context,
    looks_grounded,
)
from .retriever import Retriever

SYSTEM_PROMPT = """You are a senior system-design mentor. Answer the question using \
ONLY the provided context excerpts. Be precise and practical.

Rules:
- Ground every claim in the context. If the context is insufficient, say so explicitly.
- Prefer concrete trade-offs, numbers and examples when the context offers them.
- Structure longer answers with short headings or bullets.
- End with a "Sources" list of the file paths/URLs you drew from.
"""

GenerateFn = Callable[[str, str | None], str]


def _format_history(history: list[Turn], n: int = 6) -> str:
    return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history[-n:])


class RagPipeline:
    def __init__(self, retriever: Retriever, generate: GenerateFn, settings: Settings) -> None:
        self._retriever = retriever
        self._generate = generate
        self._s = settings

    def query(
        self, question: str, history: list[Turn] | None = None, top_k: int | None = None
    ) -> dict:
        history = history or []
        search_query = condense_query(question, history)
        hits = self._retriever.retrieve(search_query, top_k)

        # Guardrail: refuse rather than hallucinate when nothing relevant was found.
        if self._s.enable_guardrails and not has_sufficient_context(hits):
            return {
                "answer": INSUFFICIENT_CONTEXT_MESSAGE,
                "sources": [],
                "grounded": False,
                "search_query": search_query,
                "model": self._s.gemini_model,
            }

        context = "\n\n".join(
            f"--- Excerpt {i} ({h.metadata.get('source')}/{h.metadata.get('path')}) ---\n"
            f"{h.document}"
            for i, h in enumerate(hits, 1)
        )
        history_block = ""
        if history:
            history_block = f"=== CONVERSATION SO FAR ===\n{_format_history(history)}\n\n"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n{history_block}"
            f"=== CONTEXT ===\n{context}\n\n"
            f"=== CURRENT QUESTION ===\n{question}\n\n=== ANSWER ===\n"
        )
        answer = self._generate(prompt, self._s.gemini_model)

        grounded = looks_grounded(answer, hits) if self._s.enable_guardrails else True
        sources = [
            {
                "source": h.metadata.get("source", ""),
                "path": h.metadata.get("path", ""),
                "heading": h.metadata.get("heading", ""),
            }
            for h in hits
        ]
        # De-duplicate sources preserving order.
        seen, uniq = set(), []
        for s in sources:
            key = f"{s['source']}/{s['path']}"
            if key not in seen:
                seen.add(key)
                uniq.append(s)

        return {
            "answer": answer,
            "sources": uniq,
            "grounded": grounded,
            "search_query": search_query,
            "model": self._s.gemini_model,
        }
