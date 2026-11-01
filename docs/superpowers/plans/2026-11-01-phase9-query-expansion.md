# Phase 9: Optional LLM query expansion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether LLM query expansion explains qmd's NFCorpus win (0.383 vs 0.364, p = 0.02), and offer it as an opt-in setting if it helps, with its latency cost stated.

**Architecture:** `expand.py` asks a local LLM (Ollama) for `{"keywords": ..., "hypothetical": ...}`: extra search terms with synonyms, and a 1-2 sentence passage that would answer the question (HyDE). Hybrid retrieval then fuses four ranked lists with RRF: dense(query), BM25(query), dense(hypothetical), BM25(keywords). The reranker still scores against the original question. If Ollama is unavailable, search falls back to no expansion with a one-time stderr warning; search never fails because of it. Expansions are cached on disk next to the index for real use; benchmarks use no cache, so latency includes the LLM.

**Settings:** `expand_queries: bool = False`, `expand_model: str = "llama3.2:3b"`, `expand_cache: bool = True`. The Ollama client moves from `evals/llm.py` to `mcp_kb/llm.py` since search now uses it.

**Measure:** setups `hybrid-rerank20-expand` (llama3.2:3b) and `hybrid-rerank20-expand7b` (qwen2.5:7b) on SciFact, NFCorpus and FiQA; p-values vs `hybrid-rerank20` and vs qmd; p50/p95 latency.

**Decision rule (set before measuring):** the default changes only if expansion is significantly better on at least two datasets and not significantly worse on any. Otherwise it stays opt-in and the README reports the trade-off.

## Tests (fake LLM)
1. Expansion adds lists: a document matching only the hypothetical answer's wording is retrieved with expansion and not without.
2. Fallback: an unreachable Ollama gives normal results and one warning.
3. Disk cache: the second call for the same query doesn't call the LLM.
4. Bad LLM output (not JSON / missing fields) falls back to the plain query.

## Interview questions
1. What is HyDE, and why might a made-up answer retrieve better than the question?
2. Why fuse four lists instead of replacing the query?
3. Why rerank against the original question, not the expansion?
4. Why decide the "make it default" rule before seeing the numbers?
