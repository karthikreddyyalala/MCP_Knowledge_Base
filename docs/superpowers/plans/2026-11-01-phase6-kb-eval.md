# Phase 6: `kb eval` — measure search on your own docs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command that tells a user how well search works on *their* docs, which setup is best for them, and what "not found" threshold to use, with no paid API and nothing leaving the machine.

**Architecture:**

```
kb eval <docs>
  1. sync the index (same as kb index)
  2. sample N chunks (seeded; skip chunks under 25 words)
  3. local LLM (Ollama HTTP API, JSON mode) writes per chunk:
       - one question the chunk answers                       -> answerable, gold = that chunk
       - one on-topic question the chunk does NOT answer      -> candidate unanswerable
  4. filters
       - drop answerable questions whose content words mostly appear in the chunk
         (overlap > 0.75): they would reward keyword search unfairly
       - judge: the LLM sees the chunk and must agree it answers the question, else drop
       - judge: for each unanswerable candidate, the LLM sees the top 3 retrieved chunks;
         if any of them answers it, drop it (it is answerable elsewhere in the docs)
  5. save gold set: <db>/eval/gold.jsonl (reusable: kb eval --gold <file> skips the LLM)
  6. evaluate dense / bm25 / hybrid / hybrid-rerank20 (ranking only, min_score off):
       hit@5 (a top-5 result from the gold file overlapping the gold lines), MRR@5, p50 latency
       paired randomization test of each setup vs the shipped default
  7. abstention: top reranker scores for answerable vs unanswerable -> calibrate
       (balanced and answer-at-least-97%); --save-threshold writes <db>/calibration.json,
       which kb search / kb serve then use instead of the BEIR default
  8. recommendation: keep the default unless another setup is better with p < 0.05
  9. --review: show 10 random items for a human yes/no; pass rate goes in the report
```

The LLM is an injectable `generate(prompt) -> str`, so tests use a fake and run in CI without Ollama.

**Files:** `src/mcp_kb/evals/llm.py` (Ollama client, availability check), `src/mcp_kb/evals/synth.py` (sampling, prompts, filters, gold I/O), `src/mcp_kb/evals/own.py` (evaluation, abstention, recommendation, report), `src/mcp_kb/config.py` (`load_calibration`), `cli.py` (`kb eval`), `searcher.py` (results also carry `start_line`/`end_line`), tests `tests/test_synth.py`, `tests/test_own_eval.py`.

## Exit check
- Tests pass with a fake LLM (no network).
- With Ollama installed: `kb eval ./example-docs` produces a report; gold set committed for `example-docs`; review pass rate recorded.

## Interview questions
1. Why generate questions per chunk, and what bias does that introduce?
2. Why filter word overlap, and why 0.75?
3. Why use the LLM as a judge, and what are the risks of an LLM grading its own questions?
4. With only ~50 questions, how much can you trust the recommendation?
