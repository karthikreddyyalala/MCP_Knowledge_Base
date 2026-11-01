# How it works

A walkthrough of the whole system: the ideas, the data flow, every file, and every command. Read it top to bottom once; after that, use the file tour as a map.

For *why* each decision was made and what it measured, see [BUILD_LOG.md](BUILD_LOG.md).

---

## 1. The one-paragraph version

You point `kb index` at a folder of markdown files. It cuts each file into small pieces (chunks), and stores every chunk twice: once as an **embedding** (a list of 384 numbers that captures meaning) in ChromaDB, and once as **words** in a BM25 keyword index (SQLite). When Claude asks a question through MCP, `kb serve` finds the best chunks with both methods, merges the two lists (**RRF**), optionally re-scores the top 20 with a slower, more accurate model (**reranker**), and returns them with exact `file:line` citations. If even the best chunk is a poor match, it adds a "Low confidence" note so Claude doesn't present weak matches as answers (a hard "No relevant results" is available if you calibrate it on your own docs with `kb eval --save-threshold`). Everything runs on your machine; only the returned chunks enter the conversation.

---

## 2. Concepts, in the order you meet them

| Concept | Plain meaning | Where in code |
|---|---|---|
| **MCP** (Model Context Protocol) | A standard way to give Claude tools. Claude starts our program and talks to it over stdin/stdout with JSON messages. | [server.py](../src/mcp_kb/server.py) |
| **Chunk** | A piece of a document small enough to search precisely and to fit the model. | [chunking.py](../src/mcp_kb/chunking.py) |
| **Token** | The unit models read. `rollback` becomes `roll` + `##back`. Limits are in tokens, not words. | [chunking.py](../src/mcp_kb/chunking.py), [models.py](../src/mcp_kb/models.py) |
| **Max sequence length** | `all-MiniLM-L6-v2` reads at most **256 tokens** and silently drops the rest. Its tokenizer reports 512, which is a trap. | `Settings.max_tokens` in [config.py](../src/mcp_kb/config.py) |
| **Breadcrumb prefix** | `ops/runbook > Rollback` added to the start of each chunk so it "knows" where it lives. | `chunk_markdown` |
| **Embedding / dense search** | Text becomes 384 numbers; similar meanings get similar numbers. Finds paraphrases ("undo a deploy" vs "rollback"). | `_dense` in [searcher.py](../src/mcp_kb/searcher.py) |
| **Bi-encoder** | The embedding model encodes query and chunk *separately*, so chunk vectors can be precomputed. Fast. | ChromaDB + MiniLM |
| **BM25** | Classic keyword ranking: rewards words that are frequent in the chunk, rare in the collection, and doesn't let long chunks win by length. Finds exact names and codes (`ERR_4021`). | [bm25.py](../src/mcp_kb/bm25.py) |
| **FTS5** | SQLite's built-in full-text search engine. Gives us BM25 with no extra dependency. `porter` = stemming. | `bm25.write` |
| **Hybrid search** | Run dense and BM25, then merge. Each catches what the other misses. | `_retrieve` |
| **RRF** (Reciprocal Rank Fusion) | Merge ranked lists by position: each list gives an item `1 / (60 + rank)`. Uses ranks because cosine and BM25 scores are on unrelated scales. | [fusion.py](../src/mcp_kb/fusion.py) |
| **Cross-encoder / reranker** | Reads query and chunk *together* and scores the match. Much more accurate than a bi-encoder, but too slow to run on everything, so it only rescores the top candidates. | `_rerank`, `get_reranker` |
| **Low-confidence warning / abstention** | If the reranker's best score is below `warn_below` (-6.57, calibrated on BEIR), results get a warning. A hard "not found" (`min_score`) is off by default because thresholds don't transfer between kinds of text; `kb eval --save-threshold` sets it from your docs. | `Settings.warn_below`, `Settings.min_score`, [abstain.py](../src/mcp_kb/evals/abstain.py) |
| **BEIR** | A standard public benchmark suite for retrieval. We use SciFact, NFCorpus and FiQA. | [beir.py](../src/mcp_kb/evals/beir.py) |
| **qrels** | The answer key: for each query, which documents are relevant and how much (grade). | `beir.load` |
| **nDCG@10** | Are the most relevant docs at the very top? 1.0 = ideal order. The main BEIR metric. | [metrics.py](../src/mcp_kb/evals/metrics.py) |
| **Recall@10** | What share of the relevant docs made the top 10. | `recall_at_k` |
| **MRR@10** | How high the *first* correct hit is (1/rank). | `mrr_at_k` |
| **Ablation** | Turn one piece on or off and measure the change, so each piece's value is known. | `SETUPS` in `beir.py` |
| **Paired randomization test** | Is a score difference real or luck? Randomly flip the sign of each query's difference 10,000 times; if a gap as big as ours shows up often, it's noise. p < 0.05 = probably real. | [stats.py](../src/mcp_kb/evals/stats.py) |
| **Incremental sync / manifest** | Remember each file's content hash and chunk ids, so re-indexing only touches files that changed. | [indexer.py](../src/mcp_kb/indexer.py) |
| **Synthetic test set (gold set)** | Test questions written by a local LLM from your own chunks, each labelled with the chunk that answers it, plus on-topic questions your docs can't answer. | [synth.py](../src/mcp_kb/evals/synth.py) |
| **Lexical leakage** | An LLM writing a question from a passage tends to reuse its words, which makes keyword search look better than it is. Countered by the overlap filter (drop > 75% shared words) and prompt v2. | `overlap`, `ANSWERABLE_V2` |
| **LLM as judge** | The same local model checks that a question really is answered by its chunk, and that an "unanswerable" one isn't answered by the top 3 results. | `answers` in synth.py |
| **Spearman correlation** | Do two rankings agree? 1.0 = same order. Used to check `kb eval`'s ranking of setups against human-labelled BEIR. | [validate.py](../src/mcp_kb/evals/validate.py) |
| **Ollama** | Runs open LLMs locally behind an HTTP API on `localhost:11434`. Only `kb eval` uses it. | [llm.py](../src/mcp_kb/evals/llm.py) |
| **Regression gate** | A test that fails if search quality on the example docs drops. | [test_retrieval_quality.py](../tests/test_retrieval_quality.py) |

---

## 3. Data flow

### Indexing: `kb index ./docs` (incremental)

```
docs/**/*.md, *.txt
  │  indexer.sync                              src/mcp_kb/indexer.py
  ▼
scan: source = path relative to docs root, sha256 of the bytes; skip files that aren't UTF-8
compare with manifest.json (config + per-file sha256 + chunk ids)
  config changed or no manifest  -> full rebuild
  file removed or sha changed    -> delete its chunk ids from Chroma and BM25
  file new or sha changed        -> chunk it (below), upsert
  sha unchanged                  -> skip (no embedding work)
write manifest.json LAST (an interrupted sync is repaired by the next one)

for each new/changed file:
  chunking.chunk_markdown(text, source, tokenizer)
     1. split at #, ##, ### headings; keep a heading trail  [Runbook, Rollback]
     2. prefix = "ops/runbook > Runbook > Rollback\n\n"
     3. tokenize the section once, with character offsets
     4. slide a window of (256 - 2 special - prefix) tokens, 32-token overlap
     5. move window edges back to word boundaries (never split "rollback.sh")
     6. slice the ORIGINAL text by offsets; map offsets to line numbers
  ▼
chunks: {text, start_line, end_line} + source
  ids = <sha1(path)[:12]>-<n>             (stable per file, so deletes are exact)
  ├──► ChromaDB collection "kb_docs"  (embeds each chunk with all-MiniLM-L6-v2, upsert in batches)
  └──► bm25.sqlite FTS5 table          (same chunk ids, text, source, lines)
```

### Searching: Claude calls `search_docs("how do I roll back?")`

```
server.search_docs  ->  searcher.search(query, settings)          src/mcp_kb/searcher.py
  1. _retrieve
       dense: Chroma nearest 50 by embedding
       bm25:  FTS5 top 50; each query word quoted and OR-joined (punctuation can't break it)
       hybrid (default): rrf([dense ids, bm25 ids])
  2. _rerank (default on)
       cross-encoder scores (query, chunk) for the top 20 (rerank_candidates); sort by score
  3. min_score (off by default; set by kb eval --save-threshold)
       best score < min_score  ->  []  ->  "No relevant results in your docs."
  4. server.format_results: best score < warn_below (-6.57)  ->  prepend "Low confidence: ..."
  5. format top_k (5):  "[source: ops/runbook.md:L8-L14]\n<chunk text>"
```

### Measuring: `kb bench --dataset scifact --setup hybrid`

```
download BEIR zip once -> ~/.mcp-kb/bench/datasets/scifact
load corpus, test queries, qrels
write each doc as <doc_id>.md          (so the REAL indexer runs, same as for users)
build one index per chunking config    (~/.mcp-kb/bench/indexes/scifact/chroma-<hash>)
for each query: search with the setup's settings, collapse chunks -> doc ids, keep top 10
compute nDCG@10, Recall@10, MRR@10, p50/p95 latency, per-query nDCG
save results/scifact/hybrid.json, regenerate results/summary.md with Δ and p vs dense
```

### Measuring on your own docs: `kb eval ./docs`

```
sync the index
sample N chunks (seeded)
for each chunk, local LLM (qwen2.5:7b via Ollama) writes:
   answerable question     -> drop if it copies the chunk's words (overlap > 0.75)
                           -> drop unless the LLM confirms the chunk answers it
   unanswerable question   -> drop if any of the top 3 search results answers it
save gold.jsonl (reuse later with --gold, no LLM needed)
score dense / bm25 / hybrid / hybrid-rerank20: hit@5, MRR@5, p-value vs the default
calibrate "not found" on answerable vs unanswerable top scores (--save-threshold to keep it)
recommend: keep the default unless another setup is better with p < 0.05
```

`kb validate-eval --dataset nfcorpus` runs the same generation on a BEIR corpus and checks the setup ranking against human labels.

### Comparing with other tools: `kb bench --setup qmd --tool-bin <path>`

Each tool indexes the same `<doc_id>.md` files and is queried **through its own MCP server** (models load once, like real use), then scored by the same metrics code. See [adapters.py](../src/mcp_kb/evals/adapters.py).

---

## 4. File tour

### Product code: `src/mcp_kb/`

| File | What it does | Read this first |
|---|---|---|
| [config.py](../src/mcp_kb/config.py) | `Settings`: every knob in one dataclass (model, chunk size, search mode, rerank, threshold). | The defaults and their comments. |
| [chunking.py](../src/mcp_kb/chunking.py) | Heading-aware, token-budgeted chunks with exact line ranges. | `chunk_markdown`: the window loop and `word_start`. |
| [models.py](../src/mcp_kb/models.py) | Loads the tokenizer and reranker once per process (`lru_cache`). | Two short functions. |
| [indexer.py](../src/mcp_kb/indexer.py) | Incremental sync: compares the docs folder with `manifest.json` and re-indexes only changed files into ChromaDB + BM25. | The module docstring, then `sync`. |
| [bm25.py](../src/mcp_kb/bm25.py) | FTS5 keyword index: `write` and `query`. | The comment on quoting in `query`. |
| [fusion.py](../src/mcp_kb/fusion.py) | RRF in 6 lines. | The docstring. |
| [searcher.py](../src/mcp_kb/searcher.py) | The search pipeline: retrieve, rerank, abstain, format. | `search` docstring, then `_retrieve`. |
| [server.py](../src/mcp_kb/server.py) | MCP server (SDK v2 `MCPServer`) with `search_docs` and `list_docs`; syncs `--docs` and warms models at start. Logs go to stderr because stdout carries MCP. | `build_server`, `run`. |
| [cli.py](../src/mcp_kb/cli.py) | The `kb` command: `index`, `search`, `serve`, `bench`, `calibrate`. | `--help` output. |
| [\_\_init\_\_.py](../src/mcp_kb/__init__.py) | Runs first: switches Hugging Face to offline mode once models are cached, and quiets library logs. | `models_cached`. |
| [tracing.py](../src/mcp_kb/tracing.py) | Optional Arize Phoenix spans for each search. Off unless `PHOENIX_ENABLED=1`. | Skip until needed. |

### Evaluation code: `src/mcp_kb/evals/`

| File | What it does |
|---|---|
| [metrics.py](../src/mcp_kb/evals/metrics.py) | nDCG@k, Recall@k, MRR@k, matching BEIR/pytrec_eval definitions. |
| [stats.py](../src/mcp_kb/evals/stats.py) | Paired randomization test for "is this difference real?". |
| [beir.py](../src/mcp_kb/evals/beir.py) | Download, load, index, run, save, summarize; `SETUPS` defines every ablation row; `calibrate_abstention`. |
| [abstain.py](../src/mcp_kb/evals/abstain.py) | Chooses the "not found" threshold (balanced accuracy, or "answer at least 97%"). |
| [llm.py](../src/mcp_kb/evals/llm.py) | Ollama client (stdlib HTTP, JSON mode) with a clear install message if it isn't running. |
| [synth.py](../src/mcp_kb/evals/synth.py) | Writes the gold set from your chunks: prompts, overlap filter, LLM judge, save/load. |
| [own.py](../src/mcp_kb/evals/own.py) | Scores setups on the gold set, calibrates the threshold, recommends, formats the report. |
| [validate.py](../src/mcp_kb/evals/validate.py) | Checks `kb eval` against human-labelled BEIR (Spearman). |
| [adapters.py](../src/mcp_kb/evals/adapters.py) | Runs qmd and knowledge-rag through their MCP servers for the head-to-head. |

### Tests: `tests/`

| File | Protects |
|---|---|
| `test_chunking.py` | 256-token limit, exact lines, breadcrumbs, overlap, no split words. |
| `test_indexer.py` | Index builds, relative-path sources. |
| `test_sync.py` | Incremental sync: unchanged files skipped, edits replace old text, deletes disappear, same-content rewrite is unchanged, non-UTF-8 skipped, config change rebuilds, `.txt` indexed. |
| `test_searcher.py` | Citations, empty index, bm25/hybrid modes, rerank order, abstention. |
| `test_bm25.py` | Exact identifiers, stemming, hostile punctuation, empty cases. |
| `test_stats.py` | Significance test and RRF math. |
| `test_metrics.py` | Metric values computed by hand. |
| `test_abstain.py` | Threshold selection. |
| `test_beir.py` | The benchmark runner on a tiny fake dataset (no network); external-tool path mapping. |
| `test_own_eval.py` | `kb eval` with a fake LLM: every filter, de-duplication, evaluation, saved threshold, the CLI with `--gold`, and the missing-Ollama message. |
| `test_retrieval_quality.py` | Regression gate on `example-docs`: 17 answerable questions per search mode, and with shipped defaults every answerable question is answered and 5 unanswerable ones are refused. |
| `test_server.py` | Starts `kb serve` and talks real MCP to it, like Claude does; includes an edit made after indexing being picked up at startup, and an off-topic question getting "not found". |

### Other files

| Path | What it is |
|---|---|
| `results/` | Committed benchmark outputs. `summary.md` is generated; never edit it by hand. |
| `example-docs/` | Three small docs used by the regression gate and the quickstart. |
| `docs/superpowers/specs/` | The v2 design spec (goals, architecture, fairness rules). |
| `docs/superpowers/plans/` | One implementation plan per phase. |
| `.venv/` | Local Python environment (uv). Not committed. |
| `~/.mcp-kb/chroma` | Your real index (default). |
| `~/.mcp-kb/bench/` | Benchmark downloads and indexes (safe to delete; rebuilt on demand). |

---

## 5. Commands

```bash
# setup (once)
uv venv && uv pip install -e ".[dev]"

# use it
.venv/bin/kb index ./example-docs                      # build or update the index (only changed files)
.venv/bin/kb index ./example-docs --full               # rebuild from scratch
.venv/bin/kb search "how do I roll back a deployment"  # see what Claude would get
claude mcp add kb -- $PWD/.venv/bin/kb serve --docs $PWD/example-docs   # connect; re-syncs at each start

# test
.venv/bin/python -m pytest tests -q

# measure
.venv/bin/kb bench --dataset scifact --setup hybrid    # setups: see SETUPS in evals/beir.py
.venv/bin/kb calibrate                                 # "not found" threshold -> results/abstention.json
ollama serve &                                         # local LLM, only for kb eval
.venv/bin/kb eval ./example-docs                       # measure search on your own docs
.venv/bin/kb eval ./docs --gold <db>/eval/gold.jsonl   # re-score a saved test set, no LLM
.venv/bin/kb eval ./docs --save-threshold              # keep the calibrated "not found" threshold
.venv/bin/kb validate-eval --dataset nfcorpus          # check kb eval against human labels
cat results/summary.md
```

---

## 6. How to explore it yourself

1. `kb search "undo a bad deploy"` and `kb search "multipart/form-data"`: one is a paraphrase (dense wins), one an exact technical string (BM25 wins).
2. In Python, call `searcher.search(q, Settings(search_mode="dense"))`, then `"bm25"`, then `"hybrid"` and compare the citations.
3. Open `results/scifact/hybrid.json`: `per_query` shows which questions improved; compare with `dense.json`.
4. Change `max_tokens` to 128 in a bench setup and rerun; you'll see a new index folder appear and a new row in the summary.
