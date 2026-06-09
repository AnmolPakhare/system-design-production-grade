from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_provider == "local"
    assert s.gemini_model == "gemini-2.5-flash-lite"
    assert s.top_k == 6


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "11")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "gemini")
    s = Settings(_env_file=None)
    assert s.top_k == 11
    assert s.embedding_provider == "gemini"
