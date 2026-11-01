"""Retrieval regression gate: fails if search quality on example-docs drops below the v1 baseline."""
from pathlib import Path

import pytest

from mcp_kb.config import Settings
from mcp_kb.indexer import build_index
from mcp_kb.searcher import search

EXAMPLE_DOCS = Path(__file__).parent.parent / "example-docs"

# (question, file that must appear in the top 5)
GOLD = [
    # onboarding.md
    ("how do I get AWS access on day one", "onboarding.md"),
    ("what tools do I install first", "onboarding.md"),
    ("where do I find the team wiki", "onboarding.md"),
    ("who do I talk to for a code review", "onboarding.md"),
    ("what is the onboarding checklist", "onboarding.md"),
    ("how do I set up my dev environment", "onboarding.md"),
    ("what Slack channels should I join", "onboarding.md"),
    # deployment-runbook.md
    ("how do I roll back a deployment", "deployment-runbook.md"),
    ("what is the rollback command", "deployment-runbook.md"),
    ("how long does a rollback take", "deployment-runbook.md"),
    ("what Datadog threshold triggers a rollback", "deployment-runbook.md"),
    ("which Slack channel for incidents", "deployment-runbook.md"),
    ("what is the deploy checklist", "deployment-runbook.md"),
    ("how do I deploy to production", "deployment-runbook.md"),
    # api-reference.md
    ("how do I authenticate with the API", "api-reference.md"),
    ("what is the bearer token format", "api-reference.md"),
    ("how long do tokens last", "api-reference.md"),
    ("how do I refresh my token", "api-reference.md"),
    ("what does the health endpoint return", "api-reference.md"),
    ("how do I upload a document", "api-reference.md"),
]


# Recorded baselines (min Recall@5, min MRR). BM25 alone misses "what Slack channels should I join":
# the runbook mentions Slack, so keyword matching picks it over onboarding.md. Dense and hybrid get it.
BASELINES = {"dense": (1.0, 0.90), "bm25": (0.95, 0.85), "hybrid": (1.0, 0.90)}


@pytest.mark.parametrize("mode", list(BASELINES))
def test_example_docs_retrieval_does_not_regress(tmp_path, mode):
    settings = Settings(docs_path=EXAMPLE_DOCS, db_path=tmp_path / "chroma", search_mode=mode)
    build_index(settings)
    reciprocal_ranks = []
    for question, expected in GOLD:
        sources = [r["source"] for r in search(question, settings)]
        reciprocal_ranks.append(1 / (sources.index(expected) + 1) if expected in sources else 0.0)
    recall = sum(rr > 0 for rr in reciprocal_ranks) / len(GOLD)
    mrr = sum(reciprocal_ranks) / len(GOLD)
    min_recall, min_mrr = BASELINES[mode]
    assert recall >= min_recall, f"{mode}: Recall@5 dropped to {recall:.3f}"
    assert mrr >= min_mrr, f"{mode}: MRR dropped to {mrr:.3f}"
