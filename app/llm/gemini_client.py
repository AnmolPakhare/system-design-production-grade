"""Gemini SDK wrapper: generation + embeddings with quota-aware retries.

On 429 (rate limit) and 503 (transient overload) it honours the server-provided
retry delay, so long ingestion/eval runs survive free-tier quota windows.
"""
import re
import time

from ..config import Settings

_RETRY_RE = re.compile(r"retry in ([\d.]+)\s*s|retryDelay'?:?\s*'?(\d+)s", re.IGNORECASE)


def _retry_after(exc: Exception) -> float:
    m = _RETRY_RE.search(str(exc))
    if m:
        return min(float(m.group(1) or m.group(2)) + 2.0, 65.0)
    return 4.0


class GeminiClient:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        if not settings.gemini_api_key:
            raise RuntimeError("RAG_GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._settings = settings

    # --- generation ---
    def generate(self, prompt: str, model: str | None = None, max_retries: int = 6) -> str:
        model = model or self._settings.gemini_model
        delay = 3.0
        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(model=model, contents=prompt)
                return resp.text or ""
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(max(_retry_after(exc), delay))
                delay = min(delay * 2, 30.0)
        return ""

    # --- embeddings ---
    def embed(self, texts: list[str], model: str, task_type: str,
              max_retries: int = 8) -> list[list[float]]:
        from google.genai import types

        delay = 2.0
        for attempt in range(max_retries):
            try:
                resp = self._client.models.embed_content(
                    model=model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return [e.values for e in resp.embeddings]
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                time.sleep(max(_retry_after(exc), delay))
                delay = min(delay * 2, 60.0)
        return []
