# Multi-stage-ish slim image for the RAG service.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps occasionally needed by onnxruntime / chromadb.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

EXPOSE 8000

# Run via uvicorn factory (no eager app at import time).
CMD ["uvicorn", "app.service.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
