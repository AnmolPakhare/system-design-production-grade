# Architecture

`production-rag` turns a RAG prototype into a deployable service. The emphasis is
on the boundaries (config, embeddings, vector store, LLM, API) being swappable
and testable, with guardrails and observability built in.

## Layered design

```
            ┌──────────────────────── service/ ────────────────────────┐
HTTP ─────► │ FastAPI app: /health /query                              │
            │  • request-id middleware  • X-API-Key auth                │
            │  • JSON structured logs   • latency metrics               │
            └───────────────┬───────────────────────────────────────────┘
                            │ deps.build_container() (composition root)
                            ▼
            ┌──────────────────────── rag/ ────────────────────────────┐
            │ RagPipeline: condense → retrieve → guardrail → generate   │
            │   • guardrails.py  (refuse / grounding / dedup citations) │
            │   • retriever.py   (MMR diversify)                        │
            └───────┬───────────────────────────┬───────────────────────┘
                    ▼                            ▼
        embeddings/ (Embedder)           vectorstore/ (VectorStore)
        local | gemini                   chroma  (→ qdrant/pgvector…)
                    │                            ▲
                    └──── ingest/ pipeline ──────┘   llm/ GeminiClient (retries)
                          chunker + embed              (generation + judge)
                                                       eval/ Judge (LLM-as-judge)
```

## Components (`app/`)

| Module | Responsibility |
|---|---|
| `config.py` | Typed `Settings` (pydantic-settings), env-prefixed `RAG_`, cached singleton. |
| `logging_setup.py` | JSON structured logging with per-request fields. |
| `models.py` | Pydantic request/response schemas (the API contract). |
| `embeddings/` | `Embedder` interface + `local` (MiniLM) / `gemini` backends + factory. |
| `vectorstore/` | `VectorStore` interface + Chroma impl; `Hit` dataclass. |
| `llm/gemini_client.py` | Gemini generation + embeddings with 429/503 retry. |
| `ingest/chunker.py` | Pure heading-aware, section-packing chunking. |
| `ingest/pipeline.py` | Clone → chunk → embed → store (resumable, content-hash IDs). |
| `rag/retriever.py` | Diversified retrieval (MMR dedup, per-source cap). |
| `rag/guardrails.py` | Query condensing, insufficient-context refusal, grounding check. |
| `rag/pipeline.py` | Orchestrates a query; LLM injected as a callable (testable). |
| `eval/judge.py` | LLM-as-judge with model fallback + rubric parsing. |
| `service/` | FastAPI app + composition root (`deps.py`). |
| `cli.py` | `ragctl ingest|query|eval|serve`. |

## Request flow (`POST /query`)

1. Middleware assigns a `request_id`, starts a latency timer.
2. `condense_query` folds recent history into the search query.
3. `Retriever` embeds the query, pulls `top_k × multiplier` candidates, then
   diversifies (drop near-duplicates ≥ cosine threshold, cap per source).
4. **Guardrail:** if no sufficient context → return a refusal (no hallucination).
5. Build prompt (system + history + context + question) → `GeminiClient.generate`.
6. **Guardrail:** `looks_grounded` flags low context-overlap answers; sources are
   de-duplicated.
7. Response returned with `grounded`, `sources`, `model`, `request_id`,
   `latency_ms`; the query is logged as a structured event.

## Why these choices

- **Interfaces over implementations** — `Embedder` / `VectorStore` let you move
  from local MiniLM + file-Chroma to a managed embedding API + served vector DB
  without touching ingestion or retrieval.
- **Injected LLM callable** — the pipeline takes a `generate` function, so unit
  tests run fully offline with a stub (no key, no quota, deterministic).
- **Lazy LLM client** — built on first generation, so the service starts and
  `/health` works without a key, and CI needs no secrets.
- **Local embeddings default** — ingestion/serving never block on LLM quota.

## Production-readiness checklist

Implemented here:
- ✅ HTTP service, typed config, structured logging, request tracing
- ✅ Optional API-key auth, input validation (pydantic), error handling
- ✅ Guardrails (refusal + grounding signal), citation dedup
- ✅ Pluggable embeddings/vector store, resilient LLM retries
- ✅ Unit + integration + API tests (offline), CI gate, Docker/compose
- ✅ LLM-as-judge evaluation harness

Next for a real deployment (documented, not yet wired):
- ⏭️ Served/managed vector DB (Qdrant/pgvector/Pinecone) + backups
- ⏭️ Secrets manager instead of `.env`; HTTPS/ingress; horizontal autoscaling
- ⏭️ Online eval sampling, metrics export (Prometheus/OTel), dashboards, alerting
- ⏭️ Semantic response cache; model routing; per-tenant rate limits
- ⏭️ Prompt-injection defenses for crawled content; PII redaction; moderation
- ⏭️ Scheduled incremental re-ingestion with versioned, atomically-swapped indexes
- ⏭️ Golden labeled eval set + judge calibration against human labels in CI

## Testing strategy

- **Unit** — chunker, retriever MMR, guardrails, judge parsing, config.
- **Integration** — `RagPipeline` with a fake retriever + stub LLM (refusal path,
  grounded path, source dedup).
- **API** — `TestClient` against a fake container (`/health`, `/query`, auth,
  validation) — no model load, no network.
- All offline and deterministic, so the **CI gate** runs without a Gemini key.
