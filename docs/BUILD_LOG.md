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

## Phase 5: incremental sync

**Built:** `indexer.sync`: a `manifest.json` next to the index stores the chunking config and, per file, its SHA-256 and chunk ids. Sync deletes chunks of changed or removed files, chunks and upserts new or changed ones, and writes the manifest last. Chunk ids are `<sha1(path)>-<n>`, stable per file. Config changes force a full rebuild. `.txt` files are indexed; non-UTF-8 files are skipped and reported. `kb index` prints what changed (`--full` to rebuild). `kb serve --docs` syncs at startup, logging to stderr. `list_docs` shows file count, last indexed time and pending changes. BM25 store gained `add`/`delete`.

**Verified by hand:** index twice (second run: 3 unchanged, no work), edit one file and delete another (1 changed, 1 removed, 9 -> 7 chunks), the new "Database rollback" section is found at its exact line, off-topic question says "No relevant results in your docs."

**Checked that the rewrite didn't change quality:** rebuilt the SciFact index with the new code: 0.715 vs 0.720 committed. Two rebuilds differ on 1 of 300 queries, committed vs new on 3 of 300 (p = 0.25): approximate vector index noise, not a regression. README now states the noise as up to ±0.005.

**Small fix found while testing:** `kb search` said "Did you run kb index?" for off-topic questions even when an index existed; it now prints the same "not found" message as the server.

**Interview answers**
1. *Why hash content instead of modification time?* mtime changes on a copy, checkout or save-without-edits, causing needless re-embedding; it can also stay the same across real changes (restored backups). A content hash changes exactly when the bytes change.
2. *Why write the manifest last, and why upsert?* If the process dies mid-sync, the manifest still describes the old state, so the next sync redoes the work; upsert and delete-before-insert make redoing it safe.
3. *Why does a config change force a full rebuild?* A different model or token budget changes every chunk or vector, so old and new chunks would be incomparable.
4. *Why must the server log to stderr?* stdout is the MCP channel to Claude; a stray print there corrupts the protocol.

---

## Phase 6: `kb eval`, measuring search on your own docs

**Built:** `evals/llm.py` (Ollama over stdlib HTTP, JSON mode, clear install message), `evals/synth.py` (per chunk: one answerable and one on-topic unanswerable question; filters for copied wording, unsupported answers, duplicates, and "unanswerable" questions the docs answer elsewhere), `evals/own.py` (hit@5, MRR@5, p-values vs the default, threshold calibration, recommendation), `kb eval` (`--gold` reuse without an LLM, `--review` human spot-check, `--save-threshold` stored next to the index and used by `kb search`/`kb serve`), `kb validate-eval` (checks the whole thing against human-labelled BEIR). Installed Ollama with Homebrew and pulled `llama3.2:3b` and `qwen2.5:7b`.

**First real run** on example-docs worked end to end in 11 s, but showed two problems: the 25-word minimum skipped most short sections (lowered to 12), and the 3B judge accepted a question the chunk didn't really answer.

**Validation against human labels** (Spearman of the four setups' ranking; 100 chunks each):

| Model, prompt | SciFact (tuning) | NFCorpus (held out) | Answerable kept (SciFact / NFCorpus) |
|---|---|---|---|
| llama3.2:3b, v1 | 0.4 (picked bm25 as best: wrong) | not run | 28 / - |
| qwen2.5:7b, v1 | 0.8 | 0.8 | 44 / 44 |
| **qwen2.5:7b, v2 (default)** | **0.8** | **1.0** | **55 / 58** |

Prompt v2 asks for the question "the way a colleague who has NOT read it would ask", in everyday words. It was tuned on SciFact and then checked on NFCorpus, which it was never tuned on; it matched the human ranking exactly there and reduced keyword over-reward (BM25's synthetic score 0.786 -> 0.693 on NFCorpus).

**Honest limits found:**
- Synthetic questions still favour BM25 somewhat (on SciFact, BM25 ranks second on synthetic questions but third on human ones).
- "Not found" can't separate near-miss questions: on SciFact with the 7B model, keeping 98% of answerable questions refuses only 6% of on-topic unanswerable ones. The shipped threshold catches off-topic questions, not near-misses.
- Small gold sets (~50 questions) give wide error bars; the report says so and suggests `--n 100+`.

**Mistakes and fixes:** `--gold` crashed when saving the report because the `eval/` folder didn't exist yet (tests hadn't covered the CLI path; a CLI test now does). Running `knowledge-rag --help` started its server and created `documents/` and `data/` folders in the repo; they were deleted immediately.

**Interview answers**
1. *Why generate questions per chunk, and what bias does that create?* It gives every question a known answer location. The bias is lexical leakage: the model reuses the chunk's words, which flatters keyword search.
2. *Why filter word overlap at 0.75?* Questions that mostly copy the chunk test string matching, not search. 0.75 is a fixed cut-off, marked in code as one to tune if reviews show good questions being dropped.
3. *Risks of an LLM judging its own questions?* It can be lenient (the 3B model was). Mitigations: a stronger model, judging against retrieved text rather than memory, a human `--review`, and validating the whole pipeline against human labels.
4. *How far can you trust a recommendation from ~50 questions?* Only for large differences; that's why the recommendation requires p < 0.05 and otherwise keeps the default.

---

## Phase 7: head-to-head with qmd and knowledge-rag

**Built:** `evals/adapters.py`. Both tools index the same `<doc_id>.md` files and are queried **through their own MCP servers** (models load once, the way Claude uses them), then scored by the same metrics code: `kb bench --setup qmd|knowledge-rag --tool-bin <path>`. Both were installed in isolated folders (qmd's models and index kept out of `~/.cache`).

**Measured (nDCG@10):**

| | SciFact | NFCorpus | p50 latency (SciFact) |
|---|---|---|---|
| qmd 2.8.3 (LLM query expansion + LLM rerank) | 0.729 | **0.383** | 7.1 s |
| **kb** | 0.720 | 0.364 | **67 ms** |
| knowledge-rag 4.11.1 | 0.718 | 0.338 | 224 ms |

SciFact pairwise: qmd vs kb +0.008 (p = 0.62), knowledge-rag vs kb -0.003 (p = 0.78): a statistical tie. On NFCorpus, qmd wins (+0.018 over kb, p = 0.02), most likely from rewriting each question with an LLM before searching, at about 85× the latency; kb beats knowledge-rag there (+0.027, p < 0.001). That loss points at a concrete next experiment: optional LLM query expansion, measured the same way. FiQA wasn't run for the other tools (qmd would need hours to embed 57k docs here).

**Fairness notes:** qmd's Metal (GPU) shaders failed to compile on this macOS, so it likely ran on CPU and its latency isn't representative; quality is unaffected. Each tool used its documented defaults; nothing was tuned on the test set.

**Mistakes and fixes:** `knowledge-rag --help` doesn't print help, it starts the server, which created `documents/` and `data/` in the repo (deleted at once). The summary generator crashed on the new validation JSON files in `results/` (fixed, with a test). My progress logging for qmd was filtered out of the log, so its first run looked stuck; checking CPU time showed it was progressing.

---

## Phase 8: ship it

**Built:**
- **Package metadata** and a **release workflow** (`.github/workflows/release.yml`): pushing a `v*` tag runs the tests, builds, and publishes to PyPI with trusted publishing (no stored token).
- **Clean-install test:** built the wheel and installed it with `uvx` in an empty folder outside the repo: it indexed new notes, found a paraphrase, and handled an off-topic question.
- **Offline fix (found by that test):** every run was contacting the Hugging Face Hub to check for model updates, contradicting the "nothing touches the network" promise. Once all models are cached, the package now switches the Hugging Face libraries to offline mode before they load; the first run still downloads. Also silenced torch/transformers log noise. Two tests cover it.
- **Default changed from "refuse" to "warn" (found while recording the demo):** "undo a bad deployment" returned the right runbook section, but the BEIR-calibrated threshold (-6.57) refused it at score -8.3. Thresholds calibrated on one kind of text don't transfer to another. By default, results below the threshold now come with a "Low confidence" note and Claude judges; hard "not found" is opt-in through `kb eval --save-threshold` on your own docs. A test now requires three conversational paraphrases to never be hidden.
- **Demo GIF** recorded from a script (`docs/demo.tape`, rendered with `vhs`). Reviewing frames caught zsh treating `#` narration as commands and the eval report still saying "not found" after the default changed; both fixed.
- **README rewritten** for launch: demo, install, results with losses, comparison, evaluator validation, measured limits.
- **Landing page** (`docs/index.html`, deployed by Vercel from `main`) still claimed "MRR 0.585 on MS MARCO, 3.5x BM25"; replaced with the BEIR results and comparison.
- **Launch drafts:** `docs/launch/blog-post.md` and `docs/launch/posts.md` (Hacker News, LinkedIn, r/LocalLLaMA).

**Needs you:** pick a PyPI name (`mcp-knowledge-base` and `kb-mcp` are taken), choose a license, set up the PyPI trusted publisher, review the site in a browser, merge to `main`, post the launch, update the portfolio card.

---

## Remaining phases (from the spec)

| # | Phase | Status |
|---|---|---|
| 4 | Rerank + not found | done |
| 5 | Incremental sync (re-index only changed files) and sync at server start | done |
| 6 | `kb eval`: generate test questions from your own docs with a local LLM (Ollama), including unanswerable ones; recommend a setup and threshold | done |
| 7 | `kb bench` adapters for qmd and knowledge-rag; comparison table with wins and losses | done |
| 8 | PyPI + `uvx`, MCP registry, README rewrite, demo GIF, blog post, portfolio update | built; waiting on your decisions (name, license, publishing, posting) |

## How to resume

```bash
cd /Users/jaipal/claude-code/mcp-knowledge-base
git switch v2-measured-retrieval && git pull
.venv/bin/python -m pytest tests -q        # everything should pass
cat results/summary.md                     # latest numbers
```

Then open the next phase in the table above and the matching section of the spec.
