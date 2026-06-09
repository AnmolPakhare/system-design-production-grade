"""Command-line entrypoint: ingest, query, evaluate, serve."""
import argparse
import json

from .config import get_settings
from .logging_setup import configure_logging


def _cmd_ingest(args) -> None:
    from .ingest.pipeline import ingest

    settings = get_settings()
    total = ingest(settings)
    print(f"indexed {total} chunks into collection '{settings.collection}'")


def _cmd_query(args) -> None:
    from .service.deps import build_container

    container = build_container(get_settings())
    result = container.pipeline.query(args.question)
    print("\n" + result["answer"].strip() + "\n")
    print(f"grounded={result['grounded']}  sources:")
    for s in result["sources"]:
        print(f"  - {s['source']}/{s['path']}")


def _cmd_eval(args) -> None:
    from .eval.judge import DIMENSIONS, Judge
    from .service.deps import build_container

    settings = get_settings()
    container = build_container(settings)

    def gen(prompt, model=None, max_retries=6):
        from .llm.gemini_client import GeminiClient

        return GeminiClient(settings).generate(prompt, model, max_retries)

    judge = Judge(gen, [settings.judge_model, settings.gemini_model])
    questions = json.loads(args.questions) if args.questions else _DEFAULT_QS
    agg = {d: [] for d in DIMENSIONS + ["overall"]}
    for q in questions:
        r = container.pipeline.query(q)
        context = "\n\n".join(f"{s['source']}/{s['path']}" for s in r["sources"])
        v = judge.score(q, context, r["answer"])
        for d in agg:
            if isinstance(v.get(d), (int, float)):
                agg[d].append(v[d])
        print(f"overall={v.get('overall')} :: {q[:60]}")
    print("\nAVERAGES:")
    for d, vals in agg.items():
        print(f"  {d:18s}: {round(sum(vals) / len(vals), 2) if vals else 'n/a'}")


def _cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("app.service.api:create_app", factory=True, host=args.host, port=args.port)


_DEFAULT_QS = [
    "Difference between horizontal and vertical scaling and trade-offs?",
    "When should I use a message queue?",
    "What is consistent hashing and why is it used?",
]


def main() -> None:
    configure_logging(get_settings().log_level)
    p = argparse.ArgumentParser(prog="ragctl", description="Production RAG control CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Clone+chunk+embed the tutorials.").set_defaults(func=_cmd_ingest)

    q = sub.add_parser("query", help="Ask a one-off question.")
    q.add_argument("question")
    q.set_defaults(func=_cmd_query)

    e = sub.add_parser("eval", help="Run the LLM-as-judge evaluation.")
    e.add_argument("--questions", help="JSON list of questions (optional).")
    e.set_defaults(func=_cmd_eval)

    s = sub.add_parser("serve", help="Run the FastAPI service.")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
