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

**Index** - run `kb index` once. It reads your `.md` files, splits them into 400-token chunks (50-token overlap so nothing gets cut mid-sentence), embeds each chunk with `all-MiniLM-L6-v2`, and saves everything to ChromaDB on your local disk. The model downloads once from HuggingFace, then all inference is offline.

**Serve** - `kb serve` runs in the background and registers two tools with Claude Desktop over stdio: `search_docs` and `list_docs`. When you ask Claude a question, it calls `search_docs`, gets back the top 5 relevant chunks with line citations, and answers from there.

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

**400-token chunks, 50-token overlap** - big enough to capture a full thought, small enough to retrieve precisely. Overlap means nothing important gets split across a boundary.

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

Covers chunking, index building, citations, relative-path sources, the empty-index message, the MCP server end to end over stdio, and a retrieval regression gate that fails if search quality on `example-docs` drops (Recall@5 1.000, MRR 0.925 baseline).

---

## Benchmarks

Measured on [BEIR](https://github.com/beir-cellar/beir) test sets. Each document is written to a markdown file and goes through the real `kb index` and `search` code, so these numbers measure the product, not a separate script.

nDCG@10 (higher is better). `p` is a paired randomization test against the row above it; below 0.05 means the difference is unlikely to be noise.

| Setup | SciFact | NFCorpus | FiQA |
|---|---|---|---|
| v1: 400-word chunks, dense | 0.652 | 0.309 | 0.362 |
| 256-token chunks + `path > heading` prefix | 0.659 (p=0.46) | **0.323 (p=0.047)** | 0.354 (p=0.25) |
| same, prefix removed | 0.638 (p=0.07) | 0.317 (p=0.29) | 0.366 (p=0.07) |

What this shows:
- **Token-sized chunks are a correctness fix more than a score boost.** v1 chunks were ~520 tokens and the model reads 256, so half of each chunk was never searchable. BEIR documents are mostly short, so the gain is small here (significant only on NFCorpus); long documents like runbooks lose far more text under v1.
- **The prefix helps when it carries meaning.** It repeats the title on every chunk, which helps SciFact. FiQA documents have no titles, so the prefix is just a numeric file name, and it slightly hurts. On real docs the path (`ops/deployment-runbook`) is meaningful, so the prefix stays on by default.

v1 baseline matches published `all-MiniLM-L6-v2` results within about a point, which confirms the pipeline. All results with latency: [results/summary.md](results/summary.md). The v1 row was produced at commit `1016550`.

Reproduce any row:

```bash
.venv/bin/kb bench --dataset scifact --setup dense   # datasets: scifact, nfcorpus, fiqa
```
