"""Ingestion pipeline: repo markdown -> chunk -> embed -> vector store.

Resumable (content-hash IDs) so a rate-limit/quota stop never loses progress.
"""
import hashlib
import subprocess
import time
from pathlib import Path

from ..config import Settings
from ..embeddings import get_embedder
from ..logging_setup import get_logger
from ..vectorstore import get_vectorstore
from .chunker import Chunk, chunk_markdown, is_translated_readme

log = get_logger(__name__)

DEFAULT_REPOS = [
    ("system-design-primer", "https://github.com/donnemartin/system-design-primer.git"),
    ("system-design", "https://github.com/karanpratapsingh/system-design.git"),
]
EMBED_BATCH = 50
PACE_SECONDS = 32


def _chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def clone_repos(repos, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in repos:
        target = dest / name
        if target.exists():
            continue
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)


def load_repo_chunks(repo_root: Path, name: str, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md in sorted(repo_root.rglob("*.md")):
        if any(p in {".git", "node_modules"} for p in md.parts):
            continue
        if is_translated_readme(md.name):
            continue
        rel = str(md.relative_to(repo_root)).replace("\\", "/")
        raw = md.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_markdown(raw, name, rel, settings.chunk_size, settings.chunk_overlap))
    return chunks


def ingest(settings: Settings, repos=None, data_dir: Path | None = None) -> int:
    repos = repos or DEFAULT_REPOS
    data_dir = data_dir or (settings.chroma_path.parent / "repos")
    embedder = get_embedder(settings)
    store = get_vectorstore(settings)

    clone_repos(repos, data_dir)
    all_chunks: list[Chunk] = []
    for name, _ in repos:
        all_chunks.extend(load_repo_chunks(data_dir / name, name, settings))
    log.info("chunked", extra={"stage": "ingest", "n_sources": len(all_chunks)})

    existing = store.existing_ids()
    pending = [c for c in all_chunks if _chunk_id(c.text) not in existing]
    needs_pacing = getattr(embedder, "needs_pacing", False)

    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        texts = [c.text for c in batch]
        vectors = embedder.embed_documents(texts)
        store.add(
            ids=[_chunk_id(t) for t in texts],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"source": c.source, "path": c.path, "heading": c.heading} for c in batch],
        )
        if needs_pacing and start + EMBED_BATCH < len(pending):
            time.sleep(PACE_SECONDS)

    total = store.count()
    log.info("ingested", extra={"stage": "ingest", "n_sources": total})
    return total
