"""Incremental sync: only changed files are re-indexed; deletes and edits are reflected."""
import dataclasses

import pytest

from mcp_kb.config import Settings
from mcp_kb.indexer import sync
from mcp_kb.searcher import list_sources, search


@pytest.fixture
def kb(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "deploy.md").write_text("# Deploy\n\nRun kubectl apply to ship the release.\n")
    (docs / "onboarding.md").write_text("# Onboarding\n\nInstall Docker and clone the repo.\n")
    # rerank/threshold off: these tests check what is in the index, not ranking quality
    return Settings(docs_path=docs, db_path=tmp_path / "db", rerank=False, min_score=None)


def _found(settings, query, mode="hybrid"):
    return {r["source"] for r in search(query, dataclasses.replace(settings, search_mode=mode))}


def test_second_sync_does_nothing(kb):
    first = sync(kb)
    assert sorted(first.added) == ["deploy.md", "onboarding.md"] and first.full_rebuild
    second = sync(kb)
    assert (second.added, second.changed, second.removed, second.unchanged) == ([], [], [], 2)
    assert not second.full_rebuild
    assert second.total_chunks == first.total_chunks


def test_edit_replaces_old_content(kb):
    sync(kb)
    (kb.docs_path / "deploy.md").write_text("# Deploy\n\nUse the zebracorn pipeline now.\n")
    report = sync(kb)
    assert report.changed == ["deploy.md"] and report.unchanged == 1
    assert "deploy.md" in _found(kb, "zebracorn", "bm25")
    assert "deploy.md" not in _found(kb, "kubectl", "bm25")
    texts = [r["text"] for r in search("deploy pipeline", dataclasses.replace(kb, search_mode="dense"))]
    assert not any("kubectl" in t for t in texts)


def test_delete_removes_file(kb):
    sync(kb)
    (kb.docs_path / "onboarding.md").unlink()
    report = sync(kb)
    assert report.removed == ["onboarding.md"]
    assert list_sources(kb) == ["deploy.md"]
    assert "onboarding.md" not in _found(kb, "Docker")


def test_same_content_rewrite_is_unchanged(kb):
    sync(kb)
    p = kb.docs_path / "deploy.md"
    p.write_text(p.read_text())  # new mtime, same bytes
    assert sync(kb).unchanged == 2


def test_non_utf8_file_is_skipped_and_reported(kb):
    (kb.docs_path / "binary.md").write_bytes(b"\xff\xfe\x00bad")
    report = sync(kb)
    assert report.skipped == ["binary.md"]
    assert "binary.md" not in list_sources(kb)
    assert sorted(report.added) == ["deploy.md", "onboarding.md"]


def test_config_change_forces_full_rebuild(kb):
    sync(kb)
    report = sync(dataclasses.replace(kb, max_tokens=128))
    assert report.full_rebuild and sorted(report.added) == ["deploy.md", "onboarding.md"]


def test_txt_files_are_indexed(kb):
    (kb.docs_path / "notes.txt").write_text("The quarterly offsite is in Lisbon.\n")
    sync(kb)
    assert "notes.txt" in list_sources(kb)
    assert "notes.txt" in _found(kb, "Lisbon offsite")
