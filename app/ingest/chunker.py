"""Heading-aware, section-packing markdown chunking.

Splits a document at markdown headings, then packs consecutive small sections
together up to ``chunk_size`` (windowing anything larger with overlap). This
yields fewer, denser chunks than naive splitting -- better retrieval context and
far fewer embedding calls. Pure functions, no I/O, so they are easy to unit test.
"""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    text: str
    source: str
    path: str
    heading: str


def is_translated_readme(name: str) -> bool:
    """True for localized READMEs (README-ja.md, README-zh-Hans.md, ...)."""
    low = name.lower()
    return low.startswith("readme-") and low.endswith(".md")


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Return (heading_trail, body) blocks split on markdown headings."""
    blocks: list[tuple[str, str]] = []
    trail: list[str] = []
    buf: list[str] = []
    current = ""

    def flush():
        body = "\n".join(buf).strip()
        if body:
            blocks.append((current, body))

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            title = m.group(2).strip()
            trail[:] = trail[: level - 1]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(title)
            current = " > ".join(t for t in trail if t)
        else:
            buf.append(line)
    flush()
    return blocks


def window(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = start + size
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


def chunk_markdown(
    raw: str, source: str, path: str, chunk_size: int = 4000, overlap: int = 300
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf_parts: list[str] = []
    buf_len = 0
    buf_heading = ""

    def emit(body: str, heading: str):
        prefix = f"[{source} | {path}]"
        if heading:
            prefix += f" {heading}"
        chunks.append(Chunk(f"{prefix}\n\n{body}", source, path, heading))

    def flush():
        nonlocal buf_parts, buf_len, buf_heading
        body = "\n\n".join(buf_parts).strip()
        if body:
            emit(body, buf_heading)
        buf_parts, buf_len, buf_heading = [], 0, ""

    for heading, body in split_by_headings(raw):
        section = f"{heading}\n{body}" if heading else body
        if len(section) > chunk_size:
            flush()
            for w in window(section, chunk_size, overlap):
                emit(w, heading)
            continue
        if buf_len + len(section) > chunk_size:
            flush()
        if not buf_heading:
            buf_heading = heading
        buf_parts.append(section)
        buf_len += len(section)
    flush()
    return chunks
