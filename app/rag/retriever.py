"""Diversified retrieval (MMR-style).

Fetches a larger candidate pool then greedily keeps results that are not
near-duplicates of ones already kept, capping how many come from any single
source. This removes the redundant context an LLM-judge flagged on the
prototype, improving source diversity without losing relevance.
"""

import numpy as np

from ..vectorstore import Hit, VectorStore


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def diversify(
    hits: list[Hit], top_k: int, dup_threshold: float, max_per_source: int
) -> list[Hit]:
    kept: list[Hit] = []
    kept_embs: list[np.ndarray] = []
    per_source: dict = {}
    for hit in hits:
        src = hit.metadata.get("path", "")
        if per_source.get(src, 0) >= max_per_source:
            continue
        emb = np.asarray(hit.embedding, dtype=float)
        if any(_cosine(emb, ke) >= dup_threshold for ke in kept_embs):
            continue
        kept.append(hit)
        kept_embs.append(emb)
        per_source[src] = per_source.get(src, 0) + 1
        if len(kept) >= top_k:
            break
    # Backfill if diversification filtered out too much.
    if len(kept) < top_k:
        for hit in hits:
            if hit not in kept:
                kept.append(hit)
                if len(kept) >= top_k:
                    break
    return kept


class Retriever:
    def __init__(self, store: VectorStore, embedder, settings) -> None:
        self._store = store
        self._embedder = embedder
        self._s = settings

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k or self._s.top_k
        pool = max(k * self._s.candidate_multiplier, k)
        q_vec = self._embedder.embed_query(query)
        hits = self._store.query(q_vec, pool)
        return diversify(hits, k, self._s.dup_cosine_threshold, self._s.max_per_source)
