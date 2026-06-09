from app.rag.retriever import _cosine, diversify
from app.vectorstore import Hit


def _hit(path, emb, text="some document body long enough"):
    return Hit(document=text, metadata={"source": "s", "path": path}, embedding=emb)


def test_cosine_basics():
    import numpy as np

    assert abs(_cosine(np.array([1.0, 0]), np.array([1.0, 0])) - 1.0) < 1e-6
    assert abs(_cosine(np.array([1.0, 0]), np.array([0.0, 1.0]))) < 1e-6


def test_near_duplicates_dropped():
    hits = [
        _hit("a", [1.0, 0.0, 0.0]),
        _hit("b", [1.0, 0.0, 0.0]),  # identical direction -> duplicate
        _hit("c", [0.0, 1.0, 0.0]),
    ]
    kept = diversify(hits, top_k=2, dup_threshold=0.92, max_per_source=2)
    paths = [h.metadata["path"] for h in kept]
    assert paths == ["a", "c"]  # b filtered as near-duplicate of a


def test_per_source_cap():
    # top_k == 2 so the cap is tested without triggering the backfill fallback
    # (backfill deliberately relaxes constraints only to avoid starving top_k).
    hits = [
        _hit("same", [1.0, 0.0, 0.0]),
        _hit("same", [0.0, 1.0, 0.0]),
        _hit("other", [0.0, 0.0, 1.0]),
    ]
    kept = diversify(hits, top_k=2, dup_threshold=0.99, max_per_source=1)
    paths = [h.metadata["path"] for h in kept]
    assert paths.count("same") == 1
    assert "other" in paths


def test_backfill_when_over_filtered():
    hits = [_hit("a", [1.0, 0.0]), _hit("b", [1.0, 0.0])]
    kept = diversify(hits, top_k=2, dup_threshold=0.5, max_per_source=2)
    # only 'a' survives diversity, but backfill restores count to 2
    assert len(kept) == 2
