"""Lightweight, deterministic guardrails (no extra LLM call).

- condense_query: fold recent user turns into the retrieval query so follow-ups
  resolve references ("its", "that") -- free heuristic.
- has_sufficient_context: refuse to answer when retrieval returned nothing.
- looks_grounded: cheap post-hoc signal that the answer overlaps the context,
  used to flag (not block) potentially ungrounded answers.
"""
import re

from ..vectorstore import Hit

Turn = tuple[str, str]

_WORD = re.compile(r"[a-z0-9]{4,}")
_INSUFFICIENT = "I don't have enough information in the knowledge base to answer that confidently."


def condense_query(question: str, history: list[Turn], n_turns: int = 2) -> str:
    if not history:
        return question
    recent = [u for u, _ in history[-n_turns:]]
    return " ".join(recent + [question])


def has_sufficient_context(hits: list[Hit]) -> bool:
    return bool(hits) and any(len(h.document.strip()) > 50 for h in hits)


def _tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def looks_grounded(answer: str, hits: list[Hit], min_overlap: float = 0.30) -> bool:
    """Fraction of meaningful answer tokens that appear in the context."""
    ans = _tokens(answer)
    if not ans:
        return False
    ctx = set()
    for h in hits:
        ctx |= _tokens(h.document)
    overlap = len(ans & ctx) / len(ans)
    return overlap >= min_overlap


INSUFFICIENT_CONTEXT_MESSAGE = _INSUFFICIENT
