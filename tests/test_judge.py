import pytest

from app.eval.judge import DIMENSIONS, Judge, parse_verdict


def test_parse_verdict_plain_and_fenced():
    plain = '{"groundedness":5,"overall":4,"issues":[],"fixes":[]}'
    assert parse_verdict(plain)["groundedness"] == 5
    fenced = "```json\n{\"overall\": 3}\n```"
    assert parse_verdict(fenced)["overall"] == 3


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("no json here")


def test_judge_fills_missing_dimensions():
    def gen(prompt, model, max_retries):
        return '{"overall": 4}'

    v = Judge(gen, ["m1"]).score("q", "ctx", "ans")
    for d in DIMENSIONS:
        assert d in v
    assert v["_judge_model"] == "m1"


def test_judge_falls_back_on_primary_failure():
    calls = []

    def gen(prompt, model, max_retries):
        calls.append(model)
        if model == "primary":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return '{"overall": 5}'

    v = Judge(gen, ["primary", "fallback"]).score("q", "ctx", "ans")
    assert v["_judge_model"] == "fallback"
    assert calls == ["primary", "fallback"]
