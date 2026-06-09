# Running the service

Verified flow on Windows (PowerShell). On macOS/Linux, swap the venv-activate
line for `. .venv/bin/activate`.

## 1. Setup (one-time)

```powershell
cd system-design-production-grade
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## 2. Configure

```powershell
Copy-Item .env.example .env
# edit .env and set:
#   RAG_GEMINI_API_KEY=<your key>
```

Embeddings/ingestion are local and need **no key**; only answering calls Gemini.

## 3. Build the knowledge base

```powershell
ragctl ingest
# clones both tutorials, embeds locally (offline). First run downloads the
# ~80 MB MiniLM model once. Prints: indexed <N> chunks
```

## 4. Run

### As a service
```powershell
ragctl serve
# or, equivalently:
# python -m uvicorn app.service.api:create_app --factory --host 127.0.0.1 --port 8000
```

Then:
```powershell
# health
Invoke-RestMethod http://127.0.0.1:8000/health

# query
$body = '{"question":"What are the trade-offs of sharding a database?"}'
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType application/json -Body $body
```

Interactive API docs: open **http://127.0.0.1:8000/docs**.

If `RAG_API_KEY` is set in `.env`, send it as a header: `-Headers @{ "X-API-Key" = "<key>" }`.

### One-off from the CLI
```powershell
ragctl query "How does consistent hashing work?"
```

## 5. Dev / ops

```powershell
pytest                      # 25 tests, fully offline (no key/quota needed)
ruff check app tests        # lint
ragctl eval                 # LLM-as-judge evaluation (uses the judge model)
```

## Docker

```powershell
$env:RAG_GEMINI_API_KEY="<your key>"
docker compose up --build   # service on http://localhost:8000, persistent volume
```

## Notes

- `ragctl` requires the venv to be activated. Fallback: `python -m app.cli <cmd>`.
- **Generation quota:** answering uses the Gemini free-tier daily cap. `ingest`,
  `/health`, retrieval and `pytest` are local/quota-free and work anytime. A 429
  on `/query` means the daily generation quota is exhausted, not a bug.
