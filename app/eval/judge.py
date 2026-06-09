"""LLM-as-judge scoring with a model-fallback chain.

A stronger model scores (question, retrieved context, answer) on a fixed rubric
and returns concrete fixes. The preferred judge is tried first and fails fast to
a fallback when its daily quota is exhausted; the model that scored each answer
is recorded for transparency.
"""
import json
import re

DIMENSIONS = [
    "groundedness",
    "relevance",
    "completeness",
    "coherence",
    "citations",
    "retrieval_quality",
]

_TEMPLATE = """You are a strict, fair evaluator of a Retrieval-Augmented Generation \
system. Given a QUESTION, the CONTEXT it retrieved, and its ANSWER, score 1 (poor) \
to 5 (excellent):
- groundedness: every claim supported by CONTEXT; no invented facts.
- relevance: answer addresses the QUESTION.
- completeness: covers the key points CONTEXT supports.
- coherence: clear, non-redundant.
- citations: cited sources appropriate, not duplicated/noisy.
- retrieval_quality: CONTEXT contained what was needed; penalise redundant, \
off-topic or non-English context.

Respond ONLY with JSON:
{{"groundedness":n,"relevance":n,"completeness":n,"coherence":n,"citations":n,\
"retrieval_quality":n,"overall":n,"issues":["..."],"fixes":["..."]}}

=== QUESTION ===
{question}
=== CONTEXT ===
{context}
=== ANSWER ===
{answer}
"""


def parse_verdict(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]}")
    return json.loads(m.group(0))


class Judge:
    def __init__(self, generate, models: list[str]) -> None:
        self._generate = generate
        self._models = models

    def score(self, question: str, context: str, answer: str) -> dict:
        prompt = _TEMPLATE.format(question=question, context=context, answer=answer)
        last_exc = None
        for i, model in enumerate(self._models):
            is_last = i == len(self._models) - 1
            try:
                raw = self._generate(prompt, model, 6 if is_last else 1)
                verdict = parse_verdict(raw)
                verdict["_judge_model"] = model
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if is_last:
                    raise
        else:  # pragma: no cover
            raise last_exc
        for d in DIMENSIONS + ["overall"]:
            verdict.setdefault(d, None)
        verdict.setdefault("issues", [])
        verdict.setdefault("fixes", [])
        return verdict
