# Build log

What was built in each phase, why, what it measured, and what went wrong. Newest phase last. To understand the system as it is now, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) first.

**Where things live:** branch `v2-measured-retrieval` · spec: [docs/superpowers/specs/2026-10-31-kb-measured-retrieval-design.md](superpowers/specs/2026-10-31-kb-measured-retrieval-design.md) · per-phase plans: [docs/superpowers/plans/](superpowers/plans/) · numbers: [results/summary.md](../results/summary.md)

---

## Starting point (v1, before 2026-10-31)

A working prototype: index markdown into ChromaDB, two MCP tools, line citations. Problems found on review:

| Problem | Why it mattered |
|---|---|
| 400-*word* chunks (~520 tokens) | The model reads 256 tokens; half of every chunk was never searchable. |
| Citations used the filename only | `a/README.md` and `b/README.md` collided. |
| MS MARCO benchmark on a 1.1M subset, compared with a BM25 number that was never run | Inflated and not defensible ("0.585, 3.5x BM25"). |
| `mcp>=1.0` pin | A fresh install pulled SDK 2.x and the server crashed on startup. Nothing caught it. |
| Package named `src` | Only ran with `PYTHONPATH=.` from the repo folder. |
| 1,300+ lines of generated planning notes in the repo | Noise for anyone reading it. |

Research on similar tools (qmd, knowledge-rag, minirag-mcp) showed hybrid search and reranking are now the baseline. The chosen differentiator: **measure everything honestly**, and later let users measure search on their own docs (`kb eval`).

---

## Phase 0: cleanup and restructure

**Built:** real package `mcp_kb` with a `kb` command; relative-path citations; clear message when nothing is indexed; model warmup at server start; MCP server rewritten on SDK v2 (`MCPServer`) and pinned `mcp>=2.3,<3`; an end-to-end test that talks MCP over stdio; a retrieval regression gate on `example-docs`; junk removed.

**Measured:** first search after start went from 2,305 ms to 11 ms.

**Mistake corrected:** I first claimed the model reloaded on every search. Measuring showed ChromaDB caches it; only the first call was slow. Same fix (warm at start), different reason. The spec was updated.

**Lesson:** the startup crash existed in the published repo and no test noticed, because no test started the server. Test the path users actually run.

**Interview answers**
1. *What does `pip install -e .` do?* Installs the folder as a package that tracks your edits; `[project.scripts] kb = "mcp_kb.cli:app"` creates the `kb` command.
2. *Why was `from src.config` a problem?* It only worked from the repo folder with `PYTHONPATH` set. A named package works anywhere.
3. *Why warm at startup?* Same 2.3 s cost, but paid while Claude launches instead of during your first question.
4. *What does the regression gate protect?* Later phases change chunking and ranking; it fails if those changes make known questions worse.

---

## Phase 1: measure first

**Built:** metrics (nDCG@10, Recall@10, MRR@10) checked against hand-computed values; BEIR runner that writes each document to `<id>.md` and goes through the real indexer and search; `kb bench`; batched indexing (Chroma rejects very large single adds, which would have crashed on 57k files).

**Measured (v1 baseline, nDCG@10):** SciFact 0.652, NFCorpus 0.309, FiQA 0.362. Published `all-MiniLM-L6-v2` numbers are ~0.645 / ~0.316 / ~0.369, so the pipeline is sound.

**Lesson:** matching published numbers first is what makes every later number trustworthy.

**Interview answers**
1. *nDCG vs Recall vs MRR?* nDCG: best docs at the top, with graded relevance. Recall: how many relevant docs made the top 10. MRR: how high the first hit is. NFCorpus shows them disagreeing: MRR 0.50 but Recall 0.15 (first hit often right, most relevant docs missing).
2. *Why log2 discounting?* Rank 1 counts fully, rank 2 counts 63%, rank 10 counts 29%, because people read from the top.
3. *Why measure before changing?* Otherwise you can't show a change helped; it might have hurt.
4. *Why go through the real indexer?* A separate script measures something users don't run, which was the MS MARCO problem.

---

## Phase 2: token-sized chunks + breadcrumb prefix

**Built:** `chunking.py`: chunks of at most 256 tokens counted with the model's own tokenizer (including `[CLS]`/`[SEP]` and the prefix), cut only at word boundaries, text sliced from the original by character offsets so line numbers stay exact; `path > H1 > H2` prefix; bench indexes keyed by chunking config so setups coexist.

**Measured (nDCG@10, p vs previous row):**

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| v1 | 0.652 | 0.309 | 0.362 |
| 256-token + prefix | 0.659 (p=0.46) | **0.323 (p=0.047)** | 0.354 (p=0.25) |
| without prefix | 0.638 | 0.317 | 0.366 |

**Interpretation:** mostly within noise. BEIR documents are short, so v1 rarely lost text on them; the fix matters for long documents. The prefix helps when it carries meaning (paper titles) and hurts when it's a bare numeric filename (FiQA).

**Decision added to the spec:** every ablation row gets a p-value; p >= 0.05 is reported as noise.

**Interview answers**
1. *What is a tokenizer?* It splits text into the units the model reads. Limits are in tokens; one word can be several.
2. *What happens past 256 tokens?* Silently cut. The tokenizer reports 512, which makes it easy to miss.
3. *Why word boundaries and offsets?* Split words match nothing; slicing original text by offsets keeps the text and line citations exact.
4. *Did the prefix help?* Where it carried meaning, yes; where it was noise, no. Real paths are meaningful, so it's on by default.

---

## Phase 3: BM25 + RRF hybrid

**Built:** `bm25.py` (SQLite FTS5, porter stemming, every query word quoted and OR-joined); `fusion.py` (RRF, k=60); search modes `dense` / `bm25` / `hybrid`; `stats.py` (paired randomization test); `kb bench` summary now shows Δ and p vs `dense` automatically.

**Measured (nDCG@10):**

| | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| dense | 0.659 | 0.321 | 0.356 |
| BM25 | 0.685 | 0.323 | 0.241 |
| **hybrid** | **0.711** (+0.051, p<0.001) | **0.349** (+0.028, p<0.001) | 0.346 (−0.010, p=0.21) |

BM25 matches published BEIR BM25 (~0.665 / ~0.325 / ~0.236), so the keyword side is also correct.

**Decision:** hybrid is the default: significant wins on 2 of 3, no significant loss on the third. Fusion weights were *not* tuned on the test set (that would be overfitting the benchmark).

**Correction found in phase 4:** at the time I said BM25 "missed" the example question "what Slack channels should I join?" because the runbook mentions Slack. In fact the example docs never answer that question; it was a bad entry in v1's answer key, and dense only "passed" because onboarding.md was the closest file. Phase 4's "not found" threshold exposed it. With the answer key fixed, BM25 gets 17/17. The real lesson: a test set can be wrong, and a refusal mechanism is one way to find out.

**Interview answers**
1. *BM25's parts?* Term frequency (word appears often in the chunk), IDF (rare words count more), length normalization (long chunks don't win by size).
2. *Keywords vs embeddings?* Keywords win on exact names, codes, IDs; embeddings win on paraphrases.
3. *Why RRF instead of adding scores?* Cosine (0–1) and BM25 (0–30+) are on different scales; ranks are comparable.
4. *Why quote and OR?* Quotes stop punctuation and words like `AND`/`NOT` acting as FTS5 operators; OR means any matching word counts.
5. *Why a randomization test?* A mean gain can come from a few lucky queries; the test asks how often random sign flips produce a gap that large.

---

## Phase 4: cross-encoder rerank + "not found"

**Built:** `models.get_reranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded once); `searcher._rerank` rescoring the top `rerank_candidates`; `Settings.min_score`: below it, search returns nothing and Claude gets "No relevant results in your docs."; `evals/abstain.py` picks the threshold by balanced accuracy; `kb calibrate` uses cross-dataset negatives (each dataset's own queries are answerable; other datasets' queries against the same index are not).

**Before building:** relevant pair scored −0.2, irrelevant −11 (raw logits): a clear gap for a threshold. 50 pairs took ~430 ms on the Mac GPU, so reranking 20 vs 50 is an ablation row.

**Measured (nDCG@10; p vs plain hybrid):**

| | SciFact | NFCorpus | FiQA | p95 |
|---|---|---|---|---|
| hybrid | 0.711 | 0.349 | 0.346 | 31–124 ms |
| hybrid + rerank 50 | 0.709 (p=0.89) | 0.360 (p=0.06) | **0.370 (p=0.005)** | 173–280 ms |
| **hybrid + rerank 20 (default)** | 0.720 (p=0.44) | **0.364 (p=0.002)** | **0.367 (p=0.003)** | 64–325 ms |
| dense + rerank 50 | 0.704 | 0.339 | 0.372 | |

Against the original dense search, the default is +0.061 (SciFact), +0.044 (NFCorpus), both p < 0.001, and +0.011 on FiQA (p = 0.26, level). The reranker fixed FiQA, where plain hybrid had dipped. Rerank 20 beat rerank 50 on SciFact (p = 0.018) at half the cost; my guess is that more candidates give the reranker more chances to promote a wrong chunk it happens to like.

**Abstention (cross-dataset calibration, 900 answerable + 900 off-topic queries):**

| Rule | Threshold | Answered | Refused off-topic |
|---|---|---|---|
| Balanced accuracy | −3.60 | 91.9% | 83.0% |
| **Answer >= 97% (default)** | **−6.57** | **97.0%** | **69.6%** |

**Decision:** "refuse only when confident". A wrong refusal loses a real answer; a weak chunk is something Claude can judge itself.

**Mistakes and fixes along the way:**
- The first calibration used rerank 50, but the default became rerank 20, so it was rerun on the shipped setup.
- The regression gate failed with the new threshold: 3 of 20 "answerable" example questions were refused. Reading `onboarding.md` showed they are not answerable at all (no Slack, code review or wiki info). v1's answer key was wrong. They are now unanswerable test cases, and the gate checks both that real questions are answered and that unanswerable ones are refused.
- Near-miss limitation confirmed: "how do I roll back the database?" scores −4.59 and gets through. That is phase 6's job.

**Interview answers**
1. *Bi-encoder vs cross-encoder?* A bi-encoder embeds query and chunk separately, so chunk vectors are precomputed: fast but approximate. A cross-encoder reads them together, so every word of the query can attend to every word of the chunk: accurate but must run per pair at query time.
2. *Why rerank only the top N?* Cost grows linearly with N; retrieval already put most good chunks in the top 20–50.
3. *How was the threshold chosen, and its limits?* Best balanced accuracy on answerable vs off-topic queries. Off-topic questions are easier to reject than near-miss ones, so it's a starting default; `kb eval` (phase 6) recalibrates on on-topic unanswerable questions from your own docs.
4. *Why not abstain in benchmarks?* Benchmarks measure ranking quality; abstention is measured separately.

---

## Remaining phases (from the spec)

| # | Phase | Status |
|---|---|---|
| 4 | Rerank + not found | done |
| 5 | Incremental sync (re-index only changed files) and sync at server start | not started |
| 6 | `kb eval`: generate test questions from your own docs with a local LLM (Ollama), including unanswerable ones; recommend a setup and threshold | not started (needs Ollama installed) |
| 7 | `kb bench` adapters for qmd and knowledge-rag; comparison table with wins and losses | not started |
| 8 | PyPI + `uvx`, MCP registry, README rewrite, demo GIF, blog post, portfolio update | not started |

## How to resume

```bash
cd /Users/jaipal/claude-code/mcp-knowledge-base
git switch v2-measured-retrieval && git pull
.venv/bin/python -m pytest tests -q        # everything should pass
cat results/summary.md                     # latest numbers
```

Then open the next phase in the table above and the matching section of the spec.
