from app.rag.guardrails import (
    condense_query,
    has_sufficient_context,
    looks_grounded,
)
from app.vectorstore import Hit


def _hit(text):
    return Hit(document=text, metadata={"source": "s", "path": "p"}, embedding=[0.0])


def test_condense_query_no_history_passthrough():
    assert condense_query("what is sharding?", []) == "what is sharding?"


def test_condense_query_folds_history():
    history = [("Explain the Google File System architecture", "GFS has a master ...")]
    q = condense_query("what about its fault tolerance?", history)
    assert "Google File System" in q
    assert "fault tolerance" in q


def test_has_sufficient_context():
    assert not has_sufficient_context([])
    assert not has_sufficient_context([_hit("tiny")])
    assert has_sufficient_context([_hit("x" * 100)])


def test_looks_grounded():
    ctx = [_hit("Consistent hashing maps nodes onto a hash ring to minimize rebalancing")]
    grounded = "Consistent hashing uses a hash ring to minimize rebalancing of nodes"
    ungrounded = "Bananas are an excellent source of potassium and grow in tropical climates"
    assert looks_grounded(grounded, ctx)
    assert not looks_grounded(ungrounded, ctx)
