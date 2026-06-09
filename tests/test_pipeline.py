from app.config import Settings
from app.rag.pipeline import RagPipeline
from app.vectorstore import Hit


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, top_k=None):
        return self._hits


def _settings():
    return Settings(_env_file=None, gemini_model="test-model", enable_guardrails=True)


def test_insufficient_context_refuses():
    pipe = RagPipeline(FakeRetriever([]), lambda p, m=None: "should not be called", _settings())
    out = pipe.query("anything")
    assert out["grounded"] is False
    assert out["sources"] == []
    assert "enough information" in out["answer"].lower()


def test_happy_path_grounded_and_dedupes_sources():
    hits = [
        Hit("Consistent hashing maps keys to a ring to reduce rebalancing.",
            {"source": "repo", "path": "a.md", "heading": "Hashing"}, [0.1]),
        Hit("Another excerpt about the hash ring and rebalancing of nodes.",
            {"source": "repo", "path": "a.md", "heading": "Hashing"}, [0.2]),  # dup source
        Hit("Virtual nodes spread load across the ring.",
            {"source": "web", "path": "http://x", "heading": ""}, [0.3]),
    ]

    def gen(prompt, model=None):
        return "Consistent hashing uses a ring and virtual nodes to reduce rebalancing."

    out = RagPipeline(FakeRetriever(hits), gen, _settings()).query("what is consistent hashing?")
    assert out["grounded"] is True
    # a.md appears twice in hits but should be de-duplicated in sources
    keys = [f"{s['source']}/{s['path']}" for s in out["sources"]]
    assert keys.count("repo/a.md") == 1
    assert "web/http://x" in keys
