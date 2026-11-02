"""kb eval with a fake LLM: question generation, filters, evaluation, calibration (no network)."""
import dataclasses
import re

import pytest

from mcp_kb.config import Settings, save_calibration, with_calibration
from mcp_kb import llm
from mcp_kb.evals.own import evaluate
from mcp_kb.evals.synth import generate_gold, load_gold, overlap, save_gold
from mcp_kb.indexer import sync

DOCS = {
    "deploy.md": "# Deploy\n\nTo release, run kubectl apply against the production cluster from the release branch. "
                 "The pipeline waits for every pod to report healthy before it moves traffic over, and it pages "
                 "the on-call engineer if the health checks keep failing for more than ten minutes.\n",
    "onboarding.md": "# Onboarding\n\nInstall Docker Desktop and clone the repository on your first day. Ask your "
                     "manager for laptop admin rights, then run the bootstrap script, which installs Python, Node "
                     "and the pre-commit hooks that every engineer on the team is expected to use.\n",
    "billing.md": "# Billing\n\nCustomers receive an invoice on the first business day of each month. Payment is due "
                  "within thirty days, late invoices get two reminder emails, and accounts that stay unpaid for "
                  "sixty days are suspended automatically by the billing service until finance clears them.\n",
}

# fake LLM behaviour: which question it writes for each passage, and which word "proves" an answer
ANSWERABLE = {
    "kubectl": "How do we push a new version to production?",
    "Docker": "Should I install Docker Desktop and clone the repository?",  # copies the passage: overlap filter
    "invoice": "When are customers billed each month?",
}
UNANSWERABLE = {
    "kubectl": "Who approves database schema migrations?",
    "Docker": "Who approves database schema migrations?",  # duplicate: must be de-duplicated
    "invoice": "What command ships a release to production?",  # answered by deploy.md: judge must drop it
}
EVIDENCE = {
    "How do we push a new version to production?": "kubectl",
    "When are customers billed each month?": "invoice",
    "Who approves database schema migrations?": "migration",
    "What command ships a release to production?": "kubectl",
}


def fake_llm(prompt: str) -> dict:
    if "Answer with JSON only: {\"answered\"" in prompt:
        question = re.search(r"Question: (.*)", prompt).group(1).strip()
        passages = prompt.split("Passages:", 1)[1]
        return {"answered": EVIDENCE.get(question, "@@none@@") in passages}
    table = UNANSWERABLE if "does NOT answer" in prompt else ANSWERABLE
    passage = prompt.split('"""')[1]
    for word, question in table.items():
        if word in passage:
            return {"question": question}
    return {}


@pytest.fixture
def kb(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    settings = Settings(docs_path=docs, db_path=tmp_path / "db")
    sync(settings)
    return settings


def test_overlap_measures_copied_wording():
    passage = "Install Docker Desktop and clone the repository on your first day."
    assert overlap("Should I install Docker Desktop and clone the repository?", passage) > 0.75
    assert overlap("What do I set up before writing code?", passage) < 0.5


def test_generate_gold_applies_every_filter(kb):
    gold, stats = generate_gold(kb, fake_llm, n=10, seed=0)
    answerable = {g["question"]: g for g in gold if g["answerable"]}
    unanswerable = [g["question"] for g in gold if not g["answerable"]]

    assert set(answerable) == {"How do we push a new version to production?", "When are customers billed each month?"}
    assert answerable["When are customers billed each month?"]["source"] == "billing.md"
    assert unanswerable == ["Who approves database schema migrations?"]
    assert stats["dropped_overlap"] == 1  # the Docker question copied the passage
    assert stats["dropped_answered_elsewhere"] == 1  # "what command ships a release" is in deploy.md
    assert stats["dropped_duplicate"] == 1


def test_gold_round_trips(kb, tmp_path):
    gold, _ = generate_gold(kb, fake_llm, n=10, seed=0)
    save_gold(tmp_path / "gold.jsonl", gold)
    assert load_gold(tmp_path / "gold.jsonl") == gold


def test_evaluate_reports_setups_abstention_and_recommendation(kb):
    gold, _ = generate_gold(kb, fake_llm, n=10, seed=0)
    report = evaluate(kb, gold)
    assert set(report["setups"]) == {"dense", "bm25", "hybrid", "hybrid-rerank20"}
    for m in report["setups"].values():
        assert 0.0 <= m["hit@5"] <= 1.0 and 0.0 <= m["mrr@5"] <= 1.0
    assert report["setups"]["hybrid-rerank20"]["hit@5"] == 1.0
    assert report["abstention"]["n_unanswerable"] == 1
    assert "threshold" in report["abstention"]["min_answered_0.97"]
    assert report["recommendation"]["setup"] in report["setups"]


def test_saved_calibration_overrides_the_default_threshold(tmp_path):
    settings = Settings(db_path=tmp_path)
    assert with_calibration(settings).min_score == Settings().min_score
    save_calibration(tmp_path, -2.5, source="test")
    assert with_calibration(settings).min_score == -2.5


def test_missing_ollama_gives_install_instructions():
    with pytest.raises(llm.LLMUnavailable, match="ollama"):
        llm.ollama(host="http://127.0.0.1:9")  # nothing listens on port 9


def test_cli_eval_with_saved_gold_writes_report_and_threshold(tmp_path):
    from pathlib import Path

    from typer.testing import CliRunner

    from mcp_kb.cli import app

    repo = Path(__file__).parent.parent
    db = tmp_path / "db"
    result = CliRunner().invoke(app, ["eval", str(repo / "example-docs"), "--db", str(db),
                                      "--gold", str(repo / "tests/data/example_gold_handwritten.jsonl"),
                                      "--save-threshold"])
    assert result.exit_code == 0, result.output
    assert "Recommendation:" in result.output
    assert list((db / "eval").glob("report-*.json"))
    assert with_calibration(Settings(db_path=db)).min_score != Settings().min_score
