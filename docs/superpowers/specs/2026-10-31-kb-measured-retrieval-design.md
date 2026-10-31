# MCP Knowledge Base v2: Measured Local Retrieval

Date: 2026-10-31
Status: Approved

## 1. Goal

A local-first MCP server that gives Claude search over your private markdown docs, and that **measures how well it works on your docs**.

Two promises:

1. **Usable:** one command to install, one command to connect to Claude, works offline, fast, and says "not found" instead of guessing.
2. **Measured:** every retrieval technique is backed by a reproducible number on public benchmarks and on the user's own docs, compared honestly against existing tools.

Signature feature: `kb eval`, which generates a test set from the user's own docs with a local LLM and reports which search setup works best for them.

## 2. Non-goals (v1)

- PDF, DOCX, HTML or code files. Markdown and `.txt` only.
- A web UI. CLI and MCP only.
- Cloud sync, multi-user setups, auth.
- LLM-generated chunk context (Anthropic-style contextual retrieval). Too slow on CPU for benchmark-sized corpora. Revisit after v1.
- More than three MCP tools.
- Paid APIs anywhere in the runtime path.

## 3. Success criteria

| Area | Criterion |
|---|---|
| Install | A fresh Mac goes from nothing to a cited answer in Claude in under 2 minutes, using the README only |
| Speed | `search_docs` p95 under 500 ms on 1,000 chunks, CPU only, model already loaded |
| Freshness | Editing a doc and restarting the server (or running `kb index`) reflects the change without a full rebuild |
| Honesty | Every number in the README is reproducible with one documented command, and the comparison table shows losses as well as wins |
| Tests | CI runs unit tests plus a retrieval regression gate on `example-docs` |
| Learning | The author can explain every component; each phase ends with its interview questions answered in the author's own words |

## 4. Current state (what changes)

| Today | Problem | v2 |
|---|---|---|
| 400-word chunks | `all-MiniLM-L6-v2` reads 256 tokens; the rest is silently dropped | Chunks of at most 256 tokens, measured with the model's own tokenizer |
| `source = md_file.name` | Files with the same name collide | Path relative to the docs root |
| Model and DB loaded on every query | Seconds per call | Loaded once per process, warmed at server start |
| Dense search only | Misses exact names, error codes, IDs | Hybrid: dense + BM25, fused with RRF |
| Always returns 5 chunks | Irrelevant context when nothing matches | Cross-encoder rerank plus a calibrated "not found" threshold |
| Full rebuild every index | Slow on large folders | Incremental, by file content hash |
| MS MARCO eval on a 1.1M subset vs a BM25 number that was never run | Not defensible | BEIR evaluation, every system run on identical data |
| Package named `src` (`from src.config import ...`) | Not installable as a real package | Package `mcp_kb`, console command `kb` |

## 5. Architecture

### 5.1 Index flow

```
.md/.txt files
  -> file hash unchanged? skip
  -> split at headings (#, ##, ###)
  -> prefix: "path > H1 > H2" breadcrumb
  -> chunks of <= 256 tokens INCLUDING the prefix (32-token overlap), with line ranges
  -> dense: embedding -> ChromaDB
  -> sparse: text -> SQLite FTS5 (BM25, tokenizer `porter unicode61` for stemming)
  -> files table: path, sha256, chunk ids, indexed_at
```

Deleted files have their chunks removed. Changed files have their chunks replaced.

### 5.2 Query flow

```
query
  -> dense top 50  (ChromaDB)
  -> BM25 top 50   (FTS5; each query word is quoted so punctuation like `rollback.sh` or `C++` cannot break MATCH syntax)
  -> RRF fusion (k = 60)
  -> cross-encoder rerank of the fused top 50
  -> top score < threshold ? "No relevant results in your docs."
  -> top_k chunks with citation [source: path:Lstart-Lend] and rerank score
```

Each stage can be turned off, so evals can measure four setups: `dense`, `bm25`, `hybrid`, `hybrid+rerank`.

The threshold has a default calibrated on BEIR in phase 4. `kb eval` recalibrates it for the user's docs and stores it in the index's `meta` table, so `search_docs` uses it automatically.

### 5.3 Measurement flow

```
kb eval <docs>
  -> sample chunks -> local LLM writes one question per chunk (answer lives in that chunk)
  -> drop questions with high word overlap with their chunk (would unfairly favor BM25)
  -> local LLM writes on-topic questions the docs cannot answer
  -> save gold set as JSONL (reusable, reviewable, usable in CI)
  -> run all four setups -> metrics table + recommended setup + calibrated threshold

kb bench --dataset scifact|nfcorpus|fiqa [--systems kb,qmd,knowledge-rag]
  -> download BEIR dataset
  -> write each document as <doc_id>.md so every system indexes identical files
  -> run each system, map results back to doc ids by filename
     (a document split into several chunks scores as its best-ranked chunk)
  -> nDCG@10, Recall@10, MRR@10, p50/p95 latency -> results/<dataset>.json + markdown table
```

## 6. Components

Package `src/mcp_kb/`. One clear purpose per file.

| File | Purpose | Main interface |
|---|---|---|
| `config.py` | Settings dataclass; values from env vars and CLI flags | `Settings` |
| `chunking.py` | Heading-aware, token-budgeted chunking with line ranges | `chunk(text, path, tokenizer, max_tokens, overlap) -> list[Chunk]` |
| `store.py` | ChromaDB + SQLite FTS5 + files table; incremental updates | `Store.sync(docs_root) -> SyncReport`, `Store.dense(q, n)`, `Store.bm25(q, n)` |
| `search.py` | RRF, rerank, threshold; the four setups | `Searcher.search(q, mode, top_k) -> SearchResult` |
| `models.py` | Loads the embedder and reranker once, lazily | `get_embedder()`, `get_reranker()` |
| `server.py` | MCP stdio server | tools in section 7 |
| `cli.py` | Typer app: `index`, `serve`, `search`, `eval`, `bench` | `kb ...` |
| `evals/metrics.py` | Recall@k, MRR@k, nDCG@k, abstention precision and recall | pure functions |
| `evals/synth.py` | Gold-set generation through Ollama | `generate(store, n, model) -> list[GoldItem]` |
| `evals/beir.py` | BEIR download and the run loop | `run(dataset, system) -> RunResult` |
| `evals/adapters.py` | Runners for qmd and knowledge-rag via their CLIs | `QmdAdapter`, `KnowledgeRagAdapter` |
| `tracing.py` | Optional Phoenix spans; keep the current behavior, drop unused imports | `search_span()` |

Models (defaults, configurable):
- Embedder: `all-MiniLM-L6-v2` (256 tokens). `bge-small-en-v1.5` is an ablation row in `kb bench`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Gold-set LLM: a small Ollama model, default `llama3.2:3b`.

## 7. MCP tools

| Tool | Input | Output |
|---|---|---|
| `search_docs` | `query: str`, `top_k: int = 5` | Chunks with citation and score, or `No relevant results in your docs.` |
| `list_docs` | none | Indexed files, chunk count, last indexed time, files changed since |

The server syncs the index at startup when `KB_DOCS_PATH` is set, so restarting Claude Desktop picks up edits. Sync is incremental (only changed files are re-embedded), so startup stays fast after the first index.

## 8. CLI

```
kb index <docs>                 # incremental sync, prints added/changed/removed
kb serve --docs <docs>          # MCP stdio server
kb search "<query>" --explain   # shows each stage: dense rank, bm25 rank, rrf, rerank score
kb eval <docs> [--n 50] [--review]
kb bench --dataset scifact [--systems kb,qmd,knowledge-rag]
```

`kb search --explain` is the main learning and debugging tool: it shows why a chunk won.

Install and connect (target README lines):

```
uvx mcp-knowledge-base index ~/notes
claude mcp add kb -- uvx mcp-knowledge-base serve --docs ~/notes
```

The PyPI name is checked in phase 7. Fallback: `kb-mcp`.

## 9. Error handling

| Situation | Behavior |
|---|---|
| Docs path missing | CLI exits 1 with the path and a fix hint |
| Index empty or missing | `search_docs` returns `No documents indexed. Run: kb index <path>` |
| File not UTF-8 | Skipped, listed in the sync report; indexing continues |
| Ollama not running (`kb eval`) | Clear message with the install and start command; nothing else needs Ollama |
| qmd or knowledge-rag not installed (`kb bench`) | That system is skipped and marked "not run" in the table |
| Unknown tool name | MCP error, as today |

## 10. Evaluation rules (fairness)

1. Every system indexes the same files, with the same queries, on the same machine.
2. Other tools run with their documented default settings. Versions and commit hashes are recorded in each results file.
3. Raw run files are committed under `results/` so anyone can recompute the metrics.
4. The README table shows every dataset, including the ones where `kb` loses.
5. The metric code is cross-checked against `ranx` in a test.
6. The MS MARCO script and its numbers are removed, not kept alongside.

## 11. Testing

- Unit: chunking (token budget respected, line ranges correct, breadcrumb), RRF math, threshold logic, metrics against known values, incremental sync (add, change, delete).
- Integration: index `example-docs`, run `search_docs` through the MCP server over stdio.
- Regression gate in CI: a committed gold set for `example-docs`; CI fails if Recall@5 drops below the recorded baseline.
- Models are cached in CI to keep runs fast.

## 12. Build phases

Each phase runs through vibe-wise (`/vibe-wise:learn`): the author makes the design decision, Claude implements, then explains. A phase is done when its exit check passes and its interview questions are answered.

| # | Phase | Exit check | Interview questions |
|---|---|---|---|
| 0 | Cleanup and restructure: `mcp_kb` package, `kb` CLI, relative-path sources, models loaded once, delete junk (section 13) | Tests pass; second search under 500 ms | Why load models once? What does `pip install -e .` do? |
| 1 | Measure first: metrics, BEIR runner, baseline numbers for the v1 setup (400-word chunks, dense only) | Baseline table for 3 datasets committed | What are nDCG@10 and MRR? Why measure before changing anything? |
| 2 | Token-budgeted chunks plus breadcrumb prefix | Number moves vs baseline (recorded either way) | What is a tokenizer? What happens past 256 tokens? |
| 3 | BM25 (FTS5) plus RRF | Ablation row added | When does keyword search beat meaning search? Why RRF instead of adding scores? |
| 4 | Cross-encoder rerank plus "not found" threshold | Ablation row and abstention numbers | Bi-encoder vs cross-encoder? How was the threshold picked? |
| 5 | Incremental sync plus startup sync | Add, change and delete tests pass | Why hash content instead of mtime? |
| 6 | `kb eval` with Ollama gold sets | Gold set for `example-docs` committed; CI gate on | Why filter word overlap? How do you know the generated questions are good? |
| 7 | `kb bench` adapters for qmd and knowledge-rag | Comparison table with wins and losses | What makes a benchmark fair? |
| 8 | Ship: PyPI, `uvx`, MCP registry, README rewrite, demo GIF, blog post, portfolio update | A stranger installs from the README in under 2 minutes | What would you build next, and why? |

## 13. Cleanup list (phase 0)

Delete: `docs/MORNING_TEST_GUIDE.md`, the old AI-generated plans in `docs/superpowers/plans/2026-09-08-*`, tracked `src/*.egg-info/`, `docs/.DS_Store`, `.pytest_cache`, extra diagram files (keep one architecture PNG, redrawn for v2), `scripts/eval_msmarco.py`, root `index.py` and `eval.py` (replaced by `kb`).
Fix: git remote and portfolio links to `MCP_Knowledge_Base`; README badge URL.
Gitignore: `.vibe-wise/`, `.DS_Store`, `.pytest_cache/`.
Keep: this spec and the v2 implementation plan. A design doc with fairness rules is a positive signal; generic AI planning chatter is not.

## 14. Launch

- README order: demo GIF, one-line pitch, install (2 commands), "what you get back", results table with wins and losses, how it works, "why I built it this way" with links to the numbers.
- Demo GIF recorded with `vhs` (scripted, so it can be re-recorded).
- Blog post: "What actually improves local RAG: an ablation on BEIR". Post on LinkedIn, Hacker News (Show HN) and r/LocalLLaMA.
- Portfolio card: replace the MS MARCO claim with the headline ablation result and the `kb eval` feature.

## 15. Risks

| Risk | Mitigation |
|---|---|
| qmd beats `kb` on most datasets | Publish it. The differentiator is `kb eval` and the honesty, not winning every row |
| Generated questions are easy or wrong | Word-overlap filter, `--review` to spot-check 10 items, report the spot-check pass rate |
| Reranking is too slow on CPU | Rerank top 20 instead of 50; record the latency/quality trade-off as an ablation row |
| FTS5 missing from a Python build | Check at startup; fall back to `rank_bm25` with a warning |
| Other tools' output doesn't map to doc ids | One file per document named `<doc_id>.md`; map by filename |
