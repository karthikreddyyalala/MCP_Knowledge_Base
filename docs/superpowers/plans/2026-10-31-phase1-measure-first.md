# Phase 1: Measure First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Honest baseline numbers for the current v1 setup (400-word chunks, dense only) on three BEIR datasets, produced by one command: `kb bench --dataset <name>`.

**Architecture:** Metrics are pure functions. The BEIR runner downloads a dataset, writes each document to `<doc_id>.md`, indexes it with the *real* `build_index`, queries it with the *real* `search`, maps chunk hits back to document ids (a document's rank is its best chunk's rank), and saves metrics plus the raw run to `results/<dataset>/<setup>.json`. A summary table is regenerated from all result files. Later phases add setups (hybrid, rerank) by plugging in another search function; nothing else changes.

**Tech Stack:** stdlib only for download (`urllib`, `zipfile`) and metrics; existing ChromaDB and sentence-transformers.

**Datasets (BEIR test splits):** SciFact (5,183 docs, 300 queries), NFCorpus (3,633 docs, 323 queries), FiQA (57,638 docs, 648 queries). Metric definitions follow BEIR / pytrec_eval: nDCG@10 with linear gain on graded relevance, Recall@10, MRR@10.

---

## File map

| File | Responsibility |
|---|---|
| `src/mcp_kb/evals/__init__.py` | package marker |
| `src/mcp_kb/evals/metrics.py` | `ndcg_at_k`, `recall_at_k`, `mrr_at_k`, `mean` |
| `src/mcp_kb/evals/beir.py` | `download`, `load`, `write_corpus`, `run`, `write_summary` |
| `src/mcp_kb/indexer.py` | modify: add in batches (Chroma rejects very large single adds) |
| `src/mcp_kb/cli.py` | modify: `kb bench` |
| `tests/test_metrics.py` | hand-computed metric values |
| `tests/test_beir.py` | loader and runner on a tiny fake dataset (no network) |
| `results/` | committed result JSON and `summary.md` |

### Task 1: Metrics

- [ ] **Step 1: Failing tests** (`tests/test_metrics.py`)

```python
import math
from mcp_kb.evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == 1.0
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == 1.0
    assert mrr_at_k(["a", "b"], {"a": 1}, 10) == 1.0


def test_ndcg_hand_computed():
    # relevant doc at rank 2 only: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert math.isclose(ndcg_at_k(["x", "a"], {"a": 1}, 10), 1 / math.log2(3))


def test_ndcg_uses_graded_gain():
    # grades 2 and 1, returned in the wrong order
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10), dcg / idcg)


def test_cutoff_and_zero_grades_are_ignored():
    assert ndcg_at_k(["x"] * 10 + ["a"], {"a": 1}, 10) == 0.0
    assert recall_at_k(["a"], {"a": 1, "b": 0}, 10) == 1.0  # grade 0 is not relevant
    assert mrr_at_k(["x", "x", "a"], {"a": 1}, 10) == 1 / 3


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a": 1, "b": 1}, 10) == 0.5
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_metrics.py -q` -> FAIL (module missing).

- [ ] **Step 3: Implement** `src/mcp_kb/evals/metrics.py`

```python
"""Retrieval metrics, matching BEIR / pytrec_eval definitions.

ranked: doc ids in rank order (best first).
qrels:  doc id -> relevance grade for one query. Grade 0 means not relevant.
"""
import math


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrels.get(d, 0) > 0:
            return 1 / (i + 1)
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
```

- [ ] **Step 4: Run** -> PASS. **Step 5: Commit** `feat: retrieval metrics (nDCG, recall, MRR)`

### Task 2: Index in batches

- [ ] **Step 1:** In `build_index`, replace the single `col.add(...)` with:

```python
batch = client.get_max_batch_size()
for i in range(0, len(docs), batch):
    col.add(documents=docs[i:i + batch], metadatas=metas[i:i + batch], ids=ids[i:i + batch])
```

- [ ] **Step 2:** Run all tests -> PASS. **Step 3: Commit** `fix: index in batches so large doc folders don't exceed Chroma's add limit`

### Task 3: BEIR loader and runner

- [ ] **Step 1: Failing tests** (`tests/test_beir.py`) build a fake dataset on disk in BEIR format (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) with 3 documents and 2 queries, then:
  - `load(path)` returns corpus, only the queries that have test qrels, and qrels as `{qid: {doc_id: grade}}`.
  - `write_corpus(corpus, out)` writes `<doc_id>.md` files starting with `# <title>`.
  - `run(...)` on the fake dataset with the real `build_index`/`search` returns metrics with keys `ndcg@10`, `recall@10`, `mrr@10`, `p50_ms`, `p95_ms`, `n_queries`, and an obviously-answerable query ranks its document first.
  - `doc_ids_from_chunks` dedupes chunk sources to document ids in first-seen order.

- [ ] **Step 2: Run** -> FAIL.

- [ ] **Step 3: Implement** `src/mcp_kb/evals/beir.py` with:
  - `download(name, root) -> Path`: fetch `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip` once into `root`, unzip, return the dataset folder.
  - `load(path) -> (corpus, queries, qrels)`.
  - `write_corpus(corpus, out_dir)`: one file per doc; doc ids must match `[A-Za-z0-9_.-]+` (assert, so a bad id can't escape the folder).
  - `doc_ids_from_chunks(results) -> list[str]`: `Path(source).stem` deduped.
  - `run(name, corpus, queries, qrels, work_dir, setup, search_fn) -> dict`: index once per dataset (reused across setups), time each query, compute mean metrics, return metrics plus `run` (qid -> top 10 doc ids) and environment (`python`, `chromadb`, `sentence-transformers`, model name, chunk settings, date).
  - `write_summary(results_dir)`: regenerate `results/summary.md` as a table of every `results/*/*.json`.
  - Setup `v1-dense` searches with `top_k=30` chunks, so after collapsing chunks to documents at least 10 distinct docs usually remain.

- [ ] **Step 4: Run** -> PASS. **Step 5: Commit** `feat: BEIR benchmark runner`

### Task 4: `kb bench` command

- [ ] **Step 1:** `kb bench --dataset scifact|nfcorpus|fiqa [--setup v1-dense] [--work-dir ~/.mcp-kb/bench]`: download, run, save `results/<dataset>/<setup>.json`, regenerate `results/summary.md`, print the row.
- [ ] **Step 2: Smoke test:** `kb bench --dataset scifact`.
- [ ] **Step 3: Commit** `feat: kb bench command`

### Task 5: Baseline numbers

- [ ] **Step 1:** Run `kb bench` for all three datasets.
- [ ] **Step 2: Sanity check** against published numbers for `all-MiniLM-L6-v2` (SBERT/MTEB report roughly SciFact ~0.64, NFCorpus ~0.31, FiQA ~0.37 nDCG@10). A large gap means a bug in our pipeline, not a result.
- [ ] **Step 3:** Commit `results/` and add the baseline table to the README Benchmarks section. Commit `bench: v1 baseline on SciFact, NFCorpus, FiQA`.

## Exit check

- Metrics tests pass with hand-computed values.
- `results/summary.md` has a `v1-dense` row for all three datasets, reproducible with `kb bench --dataset <name>`.
- Numbers are close to published values for this model, or the gap is explained.

## Interview questions

1. What do nDCG@10, Recall@10 and MRR@10 each reward, and when would they disagree?
2. Why does nDCG use log2 discounting and graded relevance?
3. Why measure before changing anything?
4. Why index BEIR through the real `build_index` instead of embedding passages directly?
