# system-design-production-grade

A **production-shaped** Retrieval-Augmented Generation service for system-design
knowledge, powered by Google Gemini. It re-architects a working RAG prototype
into a deployable service with a typed config, pluggable embeddings/vector-store,
a FastAPI surface with guardrails and structured logging, a test suite, a CI
gate, and Docker packaging.

> Knowledge base: [system-design-primer](https://github.com/donnemartin/system-design-primer)
> and [karanpratapsingh/system-design](https://github.com/karanpratapsingh/system-design).

## Highlights

- **FastAPI service** — `/health`, `/query`; request-id tracing, optional
  `X-API-Key` auth, latency metrics, JSON structured logs.
- **Guardrails** — refuses ("I don't have enough information…") when retrieval
  finds nothing relevant; flags answers that don't overlap the context;
  de-duplicates citations.
- **Diversified retrieval (MMR)** — candidate pool → near-duplicate drop → per
  source cap, so context isn't redundant.
- **Pluggable backends** — embeddings (`local` MiniLM offline, or `gemini`) and
  vector store (Chroma; interface ready for Qdrant/pgvector/etc.).
- **Resilient LLM client** — honours 429/503 retry delays.
- **LLM-as-judge eval** — `gemini-2.5-flash` judge with fallback; 1–5 rubric.
- **Tested & CI-gated** — `pytest` (LLM mocked, fully offline), `ruff`, GitHub
  Actions (workflow in `ci/ci.yml`; move to `.github/workflows/` to activate).
  **Docker + compose** for deployment.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate      # (Windows: .venv\Scripts\Activate.ps1)
pip install -e ".[dev]"
cp .env.example .env                               # add RAG_GEMINI_API_KEY

ragctl ingest          # clone + chunk + embed the tutorials (local embeddings)
ragctl serve           # start the API at http://localhost:8000
```

Query it:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What are the trade-offs of sharding?"}'
```

## Configuration

All settings are env vars prefixed `RAG_` (see `.env.example`). Key ones:

| Var | Default | Meaning |
|---|---|---|
| `RAG_GEMINI_API_KEY` | — | Gemini key (generation/judge) |
| `RAG_GEMINI_MODEL` | `gemini-2.5-flash-lite` | generator |
| `RAG_EMBEDDING_PROVIDER` | `local` | `local` (offline) or `gemini` |
| `RAG_VECTORSTORE` | `chroma` | vector backend |
| `RAG_TOP_K` | `6` | chunks per answer |
| `RAG_API_KEY` | — | if set, clients must send `X-API-Key` |
| `RAG_ENABLE_GUARDRAILS` | `true` | groundedness/refusal guardrails |

## Testing

```bash
make test      # pytest (offline: LLM stubbed, no key/quota needed)
make lint      # ruff
make cov       # coverage report
```

## Evaluate (LLM-as-judge)

```bash
ragctl eval    # answers eval questions, scores them with the judge model
```

## Docker

```bash
docker compose up --build      # service on :8000, persistent vector volume
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design, data flow, and the
production-readiness checklist.
