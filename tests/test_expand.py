"""Optional LLM query expansion: extra lists in retrieval, safe fallback, disk cache (fake LLM)."""
import dataclasses

import pytest

from mcp_kb import expand
from mcp_kb.config import Settings
from mcp_kb.indexer import sync
from mcp_kb.searcher import search


@pytest.fixture
def kb(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "food.md").write_text("# Food\n\nThe cafeteria serves lunch from noon until two.\n")
    (docs / "money.md").write_text("# Money\n\nQuarterly revenue grew because of subscription renewals.\n")
    settings = Settings(docs_path=docs, db_path=tmp_path / "db", search_mode="bm25", rerank=False)
    sync(settings)
    return settings


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    def generator(model):
        def generate(prompt):
            calls.append(prompt)
            return {"keywords": "cafeteria lunch meal", "hypothetical": "The cafeteria serves lunch at noon."}
        return generate

    monkeypatch.setattr(expand, "_generator", generator)
    expand._cached_generator.cache_clear()
    expand._cache.cache_clear()
    yield calls
    expand._cached_generator.cache_clear()
    expand._cache.cache_clear()


def test_expansion_finds_what_the_raw_query_cannot(kb, fake_llm):
    query = "zzqx when do we eat"  # no word overlaps the food doc except common ones
    plain = [r["source"] for r in search("zzqx", kb)]
    assert plain == []
    expanded = [r["source"] for r in search("zzqx", dataclasses.replace(kb, expand_queries=True))]
    assert expanded and expanded[0] == "food.md"
    assert fake_llm and query  # the LLM was asked


def test_unreachable_llm_falls_back_to_plain_search(kb, monkeypatch, capsys):
    def broken(model):
        raise expand.llm.LLMUnavailable("ollama not running")

    monkeypatch.setattr(expand, "_generator", broken)
    expand._cached_generator.cache_clear()
    s = dataclasses.replace(kb, expand_queries=True)
    assert [r["source"] for r in search("cafeteria", s)] == ["food.md"]
    search("revenue", s)
    assert capsys.readouterr().err.count("query expansion off") == 1  # warned once
    expand._cached_generator.cache_clear()


def test_disk_cache_avoids_repeat_llm_calls(kb, fake_llm):
    s = dataclasses.replace(kb, expand_queries=True)
    search("zzqx", s)
    search("zzqx", s)
    assert len(fake_llm) == 1
    expand._cache.cache_clear()  # simulate a new process: cache must come back from disk
    search("zzqx", s)
    assert len(fake_llm) == 1


def test_bad_llm_output_is_ignored(kb, monkeypatch):
    monkeypatch.setattr(expand, "_generator", lambda model: (lambda prompt: {"oops": 1}))
    expand._cached_generator.cache_clear()
    expand._cache.cache_clear()
    s = dataclasses.replace(kb, expand_queries=True, expand_cache=False)
    assert [r["source"] for r in search("cafeteria", s)] == ["food.md"]
    expand._cached_generator.cache_clear()
