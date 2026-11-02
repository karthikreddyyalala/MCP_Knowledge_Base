from pathlib import Path
import pytest
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index






def test_build_index_returns_chunk_count(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text("# Hello\n\nThis is a test document with some content.\n")
    db_dir = tmp_path / "chroma"
    settings = Settings(docs_path=docs_dir, db_path=db_dir)
    count = build_index(settings)
    assert count >= 1






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
