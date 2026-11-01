# MCP Knowledge Base

![CI](https://github.com/karthikreddyyalala/MCP_Knowledge_Base/actions/workflows/tests.yml/badge.svg)

**[mcp-kb-site.vercel.app](https://mcp-kb-site.vercel.app)**

Give Claude Desktop semantic search over your private markdown docs. Ask a question, get a cited answer from your actual files - nothing uploaded anywhere.

---

## The problem

Your team has docs. Runbooks, onboarding guides, API references. Claude has no idea they exist.

You could upload them manually every conversation. But that sends your internal files to a third-party server, and you'd have to do it every time. This is the alternative: index them once locally, wire up the MCP server, and Claude can search them on its own whenever it needs to.

---

## How it works

<img src="https://raw.githubusercontent.com/karthikreddyyalala/MCP_Knowledge_Base/main/docs/images/architecture.png" width="720" alt="System architecture"/>

Two steps:

**Index** - run `kb index` once. It reads your `.md` files, splits them into chunks of at most 256 tokens (what the embedding model can read), prefixes each with its `path > heading`, embeds it with `all-MiniLM-L6-v2` into ChromaDB and adds it to a BM25 keyword index (SQLite FTS5), all on your local disk. The model downloads once from HuggingFace, then all inference is offline.

**Serve** - `kb serve` runs in the background and registers two tools with Claude Desktop over stdio: `search_docs` and `list_docs`. When you ask Claude a question, it calls `search_docs`, gets back the top 5 relevant chunks with line citations, and answers from there.

For the full walkthrough (every concept, file and command) see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md); for why each decision was made and what it measured, see [docs/BUILD_LOG.md](docs/BUILD_LOG.md).

The docs never leave your machine. Only the retrieved snippet (one paragraph, basically) enters the conversation.

---

## Quickstart

```bash
git clone https://github.com/karthikreddyyalala/MCP_Knowledge_Base.git
cd MCP_Knowledge_Base
uv venv && uv pip install -e .

.venv/bin/kb index ./example-docs
.venv/bin/kb search "how do I roll back a deployment"
```

`kb search` returns exactly what Claude will see, so you can check results before connecting anything.

---

## Connect to Claude

**Claude Code:**

```bash
claude mcp add kb -- /absolute/path/to/MCP_Knowledge_Base/.venv/bin/kb serve
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "/absolute/path/to/MCP_Knowledge_Base/.venv/bin/kb",
      "args": ["serve"]
    }
  }
}
```

The index lives in `~/.mcp-kb/chroma` by default. Use `--db <path>` (or `KB_DB_PATH`) on both `kb index` and `kb serve` to change it.

Restart Claude Desktop and try: *"how do I roll back a bad deployment?"*

---

## What you get back

```
[source: deployment-runbook.md:L8-L14]
If error rate spikes, run rollback: ./scripts/rollback.sh <previous-sha>
Takes ~3 minutes. Notify #incidents in Slack.
```

Line number means you can open the file and verify it. That matters more than it sounds - a system that tells you something without telling you where it came from is one you can't trust in production.

---

## Tools

| Tool | What it does |
|------|-------------|
| `search_docs(query)` | Semantic search, returns top-5 chunks with `[source: file.md:L42-L58]` |
| `list_docs()` | Lists every indexed markdown file |

---

## Tracing with Arize Phoenix

```bash
uv pip install -e ".[tracing]"
python3 -m phoenix.server.main serve   # opens http://localhost:6006
PHOENIX_ENABLED=1 .venv/bin/kb serve
```

Shows every `search_docs` call as a retrieval span - query text, which chunks were returned, citation metadata. Useful for debugging why a bad answer came back.

Used Phoenix over RAGAS because this project is a retriever, not a generator. RAGAS measures generation quality (`faithfulness`, `answer_relevancy`) which Claude handles downstream - and it needs an API key, which breaks the offline guarantee.

---

## Decisions

**ChromaDB not Pinecone** - Pinecone needs an account. ChromaDB is one pip install, stores on disk, zero setup. For a local-first tool that's the right call.

**all-MiniLM-L6-v2** - 80MB, downloads once, runs offline forever. Strong enough for English prose docs.

**256-token chunks, 32-token overlap** - the embedding model reads at most 256 tokens and silently drops the rest, so chunks are sized in real tokens with its own tokenizer. Overlap means a sentence on a boundary is still found.

**Hybrid search, fused by rank** - embeddings catch paraphrases, BM25 catches exact names and error codes. Reciprocal Rank Fusion merges them by rank position because their scores are on unrelated scales. See the benchmark table for what each piece is worth.

**Rerank only the top 20** - a cross-encoder reads the question and chunk together, which is far more accurate than comparing embeddings but too slow to run on everything. On BEIR, 20 candidates tied or beat 50 at half the cost.

**Refuse only when confident** - a wrong "not found" loses a real answer, so the threshold is set to keep 97% of answerable questions.

**Line citations not just filenames** - filenames tell you which doc. Line numbers tell you exactly where, so you can verify and notice when docs go stale.

---

## Stack

| | |
|---|---|
| MCP server | [`mcp`](https://github.com/anthropics/mcp) |
| Vector store | [`chromadb`](https://www.trychroma.com/) |
| Embeddings | `sentence-transformers` - `all-MiniLM-L6-v2` |
| Observability | [Arize Phoenix](https://phoenix.arize.com/) |
| CLI | `typer` |

---

## Tests

```bash
uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests -q
```

Covers chunking, index building, citations, relative-path sources, the empty-index message, the MCP server end to end over stdio, and a retrieval regression gate on `example-docs`: 17 answerable questions must be answered from the right file, and 5 unanswerable ones must get "not found".

---

## Benchmarks

Measured on [BEIR](https://github.com/beir-cellar/beir) test sets. Each document is written to a markdown file and goes through the real `kb index` and `search` code, so these numbers measure the product, not a separate script.

nDCG@10, higher is better. Each change is tested against the row it builds on with a paired randomization test over per-query scores; **bold** means p < 0.05, the rest may be noise.

| Setup | SciFact | NFCorpus | FiQA | p95 latency |
|---|---|---|---|---|
| v1: 400-word chunks, dense | 0.652 | 0.309 | 0.362 | 9 ms |
| 256-token chunks + `path > heading` prefix, dense | 0.659 | 0.321 | 0.356 | 8 ms |
| BM25 alone (SQLite FTS5) | 0.685 | 0.323 | 0.241 | 123 ms |
| Hybrid: dense + BM25, RRF fusion | **0.711** (+0.051) | **0.349** (+0.028) | 0.346 (−0.010) | 124 ms |
| **Hybrid + cross-encoder rerank of top 20 (default)** | 0.720 (+0.010) | **0.364** (+0.016) | **0.367** (+0.021) | 65–325 ms |

What this shows:
- **Hybrid + rerank is the default.** Against the original dense search it is +0.061 on SciFact and +0.044 on NFCorpus (both p < 0.001) and level on FiQA (+0.011, p = 0.26).
- **Hybrid alone dipped on FiQA** (paraphrased finance questions with little word overlap, where BM25 is weak at 0.241). Fusion weights were not tuned on the test set, since that would overfit the benchmark. The reranker fixed it instead: +0.021 over hybrid on FiQA (p = 0.003).
- **Reranking 20 candidates tied or beat reranking 50** (SciFact +0.012, p = 0.018) at about half the latency, so 20 is the default.
- **Sanity checks:** dense matches published `all-MiniLM-L6-v2` results and BM25 matches published BM25 baselines (SciFact ~0.665, NFCorpus ~0.325, FiQA ~0.236) within about a point.
- **Token-sized chunks are a correctness fix more than a score boost.** v1 chunks were ~520 tokens and the model reads 256, so half of each chunk was never searchable. BEIR documents are short, so the effect is small here; long documents like runbooks lose far more text under v1.
- **The context prefix** helps where it carries meaning (paper titles on SciFact) and slightly hurts where it is a bare numeric file name (FiQA, p = 0.048). Real file paths are meaningful, so it stays on.
- Dense scores vary by about ±0.002 between index rebuilds because the vector index is approximate (HNSW).

Full table with Recall@10, MRR@10, latency and p-values: [results/summary.md](results/summary.md). The v1 row was produced at commit `1016550`.

### "Not found" instead of guessing

When even the best reranked chunk scores below a threshold, `search_docs` answers "No relevant results in your docs." instead of handing Claude unrelated text. BEIR has no unanswerable questions, so the threshold is calibrated with cross-dataset negatives: each dataset's own queries are answerable, and the other datasets' queries asked of the same corpus (finance questions to a science library) are not. The default favours answering, because a wrong refusal loses a real answer while a weak chunk is something Claude can judge for itself:

| Threshold rule | Answerable questions answered | Off-topic questions refused |
|---|---|---|
| Balanced accuracy | 91.9% | 83.0% |
| **Answer at least 97% (default, -6.57)** | **97.0%** | **69.6%** |

Limits, stated plainly: off-topic questions are easier to reject than near-misses. "How do I roll back the database?" against docs that only cover app rollbacks still gets an answer. `kb eval` (in progress) will recalibrate on on-topic unanswerable questions generated from your own docs. Raw scores: [results/abstention.json](results/abstention.json).

The threshold also caught a bug in this repo's own test set: three "answerable" example questions (Slack channels, code review, team wiki) are not answered anywhere in `example-docs`. They are now tested as unanswerable.

Reproduce any row:

```bash
.venv/bin/kb bench --dataset scifact --setup hybrid-rerank20   # setups: see SETUPS in src/mcp_kb/evals/beir.py
.venv/bin/kb calibrate                               # not-found threshold
```
