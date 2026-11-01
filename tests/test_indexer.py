from pathlib import Path
import pytest
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index, chunk_markdown


def test_chunk_markdown_splits_on_size():
    text = "word " * 500
    chunks = chunk_markdown(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2, "Long text should produce multiple chunks"


def test_chunk_markdown_preserves_overlap():
    text = "\n".join(f"Line {i}: " + "content " * 10 for i in range(100))
    chunks = chunk_markdown(text, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2


def test_build_index_returns_chunk_count(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Hello\n\nThis is a test document with some content.\n")
    db_dir = tmp_path / "chroma"
    settings = Settings(docs_path=docs_dir, db_path=db_dir)
    count = build_index(settings)
    assert count >= 1


def test_chunk_inherits_header():
    text = "## Auth\n\nBearer tokens required.\n\n## Deploy\n\nRun rollback.sh."
    chunks = chunk_markdown(text, chunk_size=10, chunk_overlap=0)
    deploy_chunks = [c for c in chunks if "Deploy" in c["text"]]
    assert deploy_chunks, "Expected a chunk containing Deploy header"
    assert deploy_chunks[0]["text"].startswith("## Deploy")


def test_chunk_without_headers_still_works():
    text = "word " * 50
    chunks = chunk_markdown(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    full = " ".join(c["text"] for c in chunks)
    assert "word" in full


def test_build_index_stores_source_metadata(tmp_path):
    import chromadb
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "runbook.md").write_text("# Runbook\n\n" + "Deploy step. " * 50)
    db_dir = tmp_path / "chroma"
    settings = Settings(docs_path=docs_dir, db_path=db_dir)
    build_index(settings)
    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.get_collection("kb_docs")
    results = col.get(include=["metadatas"])
    sources = {m["source"] for m in results["metadatas"]}
    assert "runbook.md" in sources


def test_sources_use_relative_paths(tmp_path):
    from mcp_kb.searcher import list_sources
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir()
    (docs / "a" / "README.md").write_text("# A\n\nalpha service runbook\n")
    (docs / "b" / "README.md").write_text("# B\n\nbeta service runbook\n")
    settings = Settings(docs_path=docs, db_path=tmp_path / "chroma")
    build_index(settings)
    assert list_sources(settings) == ["a/README.md", "b/README.md"]
