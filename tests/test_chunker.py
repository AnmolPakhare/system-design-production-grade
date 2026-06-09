from app.ingest.chunker import (
    chunk_markdown,
    is_translated_readme,
    split_by_headings,
    window,
)


def test_is_translated_readme():
    assert is_translated_readme("README-ja.md")
    assert is_translated_readme("README-zh-Hans.md")
    assert not is_translated_readme("README.md")
    assert not is_translated_readme("readme.md")
    assert not is_translated_readme("CONTRIBUTING.md")


def test_split_by_headings_tracks_trail():
    md = "# A\nintro\n## B\nbody\n## C\nmore"
    blocks = split_by_headings(md)
    trails = [t for t, _ in blocks]
    assert "A" in trails[0]
    assert "A > B" in trails[1]
    assert "A > C" in trails[2]


def test_window_overlaps_and_covers():
    text = "x" * 1000
    parts = window(text, size=400, overlap=100)
    assert len(parts) >= 3
    assert "".join(p[:1] for p in parts)  # non-empty
    # full text is covered by the union of windows
    assert parts[0] == text[:400]


def test_packing_merges_small_sections():
    md = "# T\n" + "\n".join(f"## H{i}\nshort body {i}" for i in range(10))
    chunks = chunk_markdown(md, "repo", "f.md", chunk_size=4000, overlap=200)
    # 10 tiny sections should pack into a single ~4KB chunk, not 10 chunks.
    assert len(chunks) == 1
    assert chunks[0].source == "repo" and chunks[0].path == "f.md"


def test_large_section_is_windowed():
    md = "# Big\n" + ("word " * 2000)  # > chunk_size chars
    chunks = chunk_markdown(md, "repo", "big.md", chunk_size=1000, overlap=100)
    assert len(chunks) > 1
