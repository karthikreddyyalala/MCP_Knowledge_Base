# Phase 2: Token-Budgeted Chunks with Context Prefix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every chunk fits what the embedding model actually reads (256 tokens including special tokens and the prefix), and each chunk carries a `path > H1 > H2` breadcrumb. Measure the effect on BEIR, with the prefix ablated separately.

**Architecture:** `chunking.py` tokenizes each heading section once with the model's own fast tokenizer (`return_offsets_mapping`), slides a token window over it, snaps window edges to word boundaries (never start or end inside a `##` WordPiece), and slices the *original* text by character offsets, so chunk text is exact and line ranges come from offsets. `models.get_tokenizer` loads the tokenizer once. Bench setups become `name -> (settings overrides, search fn)`, and each index lives in a folder keyed by its chunking config, so setups with different chunking coexist.

**Tech Stack:** `transformers` fast tokenizer (already installed with sentence-transformers).

**Before:** v1 chunks are 400 words (~520 tokens); the model drops everything past 256. Baseline nDCG@10: SciFact 0.652, NFCorpus 0.309, FiQA 0.362.

---

### Task 1: Tokenizer loader
- `src/mcp_kb/models.py`: `get_tokenizer(model_name)` with `functools.lru_cache`; bare names map to `sentence-transformers/<name>`.

### Task 2: Chunker (TDD) — `src/mcp_kb/chunking.py`, `tests/test_chunking.py`
Tests (real tokenizer):
1. Every chunk of a 2,000-word section encodes to <= 256 tokens *with* special tokens.
2. A unique word on line N lands in a chunk whose `start_line <= N <= end_line`.
3. `## Deploy` under `# Runbook` in `ops/runbook.md` -> chunk text starts with `ops/runbook > Runbook > Deploy`.
4. Consecutive chunks of a long section overlap (the start of chunk i+1 appears in chunk i).
5. No chunk starts or ends mid-word.
6. Empty text -> `[]`; text with no headings -> prefix is just the path.
7. `context_prefix=False` -> no breadcrumb.

Interface:
```python
def chunk_markdown(text: str, source: str, tokenizer, max_tokens: int = 256,
                   overlap: int = 32, context_prefix: bool = True) -> list[dict]:
    """Returns [{"text", "start_line", "end_line"}]."""
```

### Task 3: Wire into the indexer and settings
- `Settings`: `chunk_size`/`chunk_overlap` (words) -> `max_tokens=256`, `overlap_tokens=32`, `context_prefix=True`.
- `indexer.build_index` calls `chunking.chunk_markdown(text, source, get_tokenizer(...), ...)`; the old word chunker is deleted.
- Old chunker tests move to the new interface; the regression gate must still pass (Recall@5 1.000, MRR >= 0.90).

### Task 4: Bench setups and per-config index folders
- `SETUPS = {"dense": ({}, search), "dense-no-prefix": ({"context_prefix": False}, search)}`.
- Index folder: `<work>/<dataset>/<sha1 of chunk config>[:10]`.
- The v1 rows stay in `results/` as history; README notes they were produced at commit `1016550`.

### Task 5: Measure
- `kb bench` with `dense` and `dense-no-prefix` on all three datasets; update README table; commit.

## Exit check
- All tests pass; no chunk exceeds 256 tokens.
- Ablation rows recorded for both setups on all three datasets, whatever the result.

## Interview questions
1. What is a tokenizer, and why count tokens instead of words?
2. What happens to text past the model's max sequence length?
3. Why snap chunk edges to word boundaries, and why slice original text by offsets?
4. Did the prefix help? On which dataset, and why might it help real docs more than BEIR?
