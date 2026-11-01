import re
import pytest
from pathlib import Path
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index
from mcp_kb.searcher import search, list_sources


@pytest.fixture
def indexed_settings(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "deploy.md").write_text(
        "# Deployment\n\n"
        "To deploy the app run: kubectl apply -f k8s/\n\n"
        "Check pod status with: kubectl get pods\n\n" * 20
    )
    (docs_dir / "onboarding.md").write_text(
        "# Onboarding\n\nClone the repo and run npm install.\n\n" * 20
    )
    settings = Settings(docs_path=docs_dir, db_path=tmp_path / "chroma")
    build_index(settings)
    return settings


def test_search_returns_results(indexed_settings):
    results = search("how to deploy", indexed_settings)
    assert len(results) >= 1


def test_search_result_has_citation(indexed_settings):
    results = search("kubectl deployment", indexed_settings)
    assert results[0]["citation"].startswith("[source:")
    assert "deploy.md" in results[0]["citation"]
    assert ":L" in results[0]["citation"]


def test_search_citation_format(indexed_settings):
    results = search("deploy", indexed_settings)
    citation = results[0]["citation"]
    assert re.match(r"\[source: .+\.md:L\d+-L\d+\]", citation)


def test_list_sources_returns_filenames(indexed_settings):
    sources = list_sources(indexed_settings)
    assert "deploy.md" in sources
    assert "onboarding.md" in sources


def test_empty_index_returns_nothing(tmp_path):
    settings = Settings(db_path=tmp_path / "empty")
    assert search("anything", settings) == []
    assert list_sources(settings) == []
