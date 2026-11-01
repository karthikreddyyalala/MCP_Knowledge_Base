# Phase 4: Cross-Encoder Rerank + "Not Found"

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerank fused candidates with a local cross-encoder, and answer "No relevant results in your docs." when the best reranked chunk scores below a calibrated threshold.

**Architecture:** `searcher.search` first retrieves candidates (dense, bm25 or hybrid). If `settings.rerank` is on, it scores `(query, chunk)` pairs for the top `rerank_candidates` with `cross-encoder/ms-marco-MiniLM-L-6-v2` (loaded once in `models.get_reranker`), sorts by score, and attaches `score` to each result. If `settings.min_score` is set and the best score is below it, search returns `[]` and the server says "No relevant results in your docs." Bench setups never abstain (they measure ranking).

**Calibration without labeled unanswerable questions:** BEIR has none, so `kb calibrate` uses cross-dataset negatives: each dataset's own queries (answerable) vs a sample of the other datasets' queries run against the same index (unanswerable, e.g. finance questions against a science corpus). It picks the threshold with the best balanced accuracy and saves `results/abstention.json`. Off-topic negatives are easier than near-miss ones, so this is a starting default; phase 6 (`kb eval`) recalibrates on on-topic unanswerable questions from the user's own docs.

**Measured before building:** relevant pair -0.2, irrelevant -11 (logits). 50 pairs ~430 ms on MPS, so `rerank_candidates` 20 vs 50 is an ablation row.

## Tasks
1. `models.get_reranker` (lru_cache).
2. TDD `searcher`: rerank puts the rollback chunk first with descending `score`; `min_score=100` returns `[]`; `min_score=None` never abstains.
3. TDD `evals/abstain.py`: `calibrate(pos, neg) -> {"threshold", "balanced_accuracy", "answered_rate", "abstained_rate"}`; perfectly separable scores give balanced accuracy 1.0 with a threshold between them.
4. Server: indexed but `[]` -> "No relevant results in your docs."; stdio test with an off-topic query.
5. Bench setups `hybrid-rerank` (50) and `hybrid-rerank20`, plus `dense-rerank` (does reranking alone fix FiQA?); `kb calibrate`; set defaults from the numbers; README; commit.

## Exit check
- Rerank rows with p-values on all three datasets; p95 latency recorded.
- `results/abstention.json` with threshold and balanced accuracy; an off-topic question to the example docs returns "No relevant results".

## Interview questions
1. Bi-encoder vs cross-encoder: why is one fast and the other accurate?
2. Why rerank only the top N, and how did you pick N?
3. How was the "not found" threshold chosen, and what are its limits?
4. Why not abstain during benchmarks?
