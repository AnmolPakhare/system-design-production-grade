from fastapi.testclient import TestClient

from app.config import Settings
from app.service.api import create_app
from app.service.deps import Container


class FakeStore:
    def count(self):
        return 42


class FakePipeline:
    def query(self, question, history=None, top_k=None):
        return {
            "answer": "A grounded answer.",
            "sources": [{"source": "repo", "path": "a.md", "heading": "H"}],
            "grounded": True,
            "search_query": question,
            "model": "test-model",
        }


def _client(api_key=""):
    settings = Settings(_env_file=None, api_key=api_key, gemini_model="test-model")
    container = Container(settings=settings, store=FakeStore(), pipeline=FakePipeline())
    return TestClient(create_app(settings, container))


def test_root_serves_chat_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "System Design RAG" in r.text


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["collection_count"] == 42
    assert body["generator_model"] == "test-model"


def test_query_ok_and_request_id_header():
    r = _client().post("/query", json={"question": "what is sharding?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "A grounded answer."
    assert body["grounded"] is True
    assert body["sources"][0]["path"] == "a.md"
    assert "X-Request-ID" in r.headers
    assert body["latency_ms"] >= 0


def test_query_validation_error_on_empty_question():
    r = _client().post("/query", json={"question": ""})
    assert r.status_code == 422


def test_api_key_enforced_when_configured():
    c = _client(api_key="secret")
    assert c.post("/query", json={"question": "hi there"}).status_code == 401
    ok = c.post("/query", json={"question": "hi there"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
