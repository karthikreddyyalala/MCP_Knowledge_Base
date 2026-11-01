# Phase 0: Cleanup and Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the prototype into a real installable package (`mcp_kb`, command `kb`) with path-safe citations, a warm model at server start, a retrieval regression test, and no junk files.

**Architecture:** Same retrieval behavior as v1 (dense search, 400-word chunks). Phase 0 changes structure, not quality, so the regression test must still show Recall@5 = 1.000 and MRR >= 0.90 on `example-docs`. Quality changes start in phase 2, after phase 1 measures the baseline.

**Tech Stack:** Python 3.12 (uv venv), ChromaDB 1.5, sentence-transformers 6.2, MCP Python SDK, Typer, pytest + pytest-asyncio.

**Baseline before this phase (measured 2026-10-31):** 10 tests pass. `example-docs`: Recall@5 1.000 (20/20), MRR 0.925, 9 chunks. Search: first call 2,305 ms, later calls 5-8 ms (ChromaDB caches the model in-process, so the cost is a cold start, not a per-call reload).

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/mcp_kb/__init__.py` | create (move from `src/__init__.py`) | package marker |
| `src/mcp_kb/config.py` | move | `Settings` |
| `src/mcp_kb/indexer.py` | move + modify | chunking and index build; relative-path sources |
| `src/mcp_kb/searcher.py` | move + modify | search, list sources, empty-index handling, warmup |
| `src/mcp_kb/server.py` | move + rewrite | MCP server built from `Settings`; warmup at start |
| `src/mcp_kb/tracing.py` | move + modify | drop unused imports |
| `src/mcp_kb/cli.py` | create | `kb index`, `kb search`, `kb serve` |
| `tests/test_retrieval_quality.py` | create | regression gate on `example-docs` |
| `tests/test_server.py` | create | MCP over stdio, end to end |
| `index.py`, `eval.py`, `scripts/eval_msmarco.py` | delete | replaced by `kb` |
| junk (Task 1) | delete | |

---

### Task 1: Remove junk and fix repo hygiene

**Files:** delete listed files; modify `.gitignore`, `pyproject.toml`

- [ ] **Step 1: Delete files**

```bash
git rm -q docs/MORNING_TEST_GUIDE.md docs/superpowers/plans/2026-09-08-*.md \
  docs/images/system-overview.svg -r docs/diagrams scripts/eval_msmarco.py
git rm -q -r --cached src/mcp_knowledge_base.egg-info && rm -rf src/mcp_knowledge_base.egg-info
rm -f docs/.DS_Store && rm -rf .pytest_cache
```

- [ ] **Step 2: Extend `.gitignore`**

Append:
```
.DS_Store
.pytest_cache/
.vibe-wise/
```

- [ ] **Step 3: Remove the `benchmark` extra from `pyproject.toml`** (MS MARCO script is gone; phase 1 adds its own).

- [ ] **Step 4: Point the remote at the renamed repo**

```bash
git remote set-url origin https://github.com/karthikreddyyalala/MCP_Knowledge_Base.git
```

- [ ] **Step 5: Commit** `chore: remove stale docs, build artifacts and MS MARCO script`

### Task 2: Make `mcp_kb` a real package with a `kb` command

**Files:** move `src/*.py` to `src/mcp_kb/`; modify `pyproject.toml`, tests

- [ ] **Step 1: Move files**

```bash
mkdir -p src/mcp_kb
git mv src/__init__.py src/config.py src/indexer.py src/searcher.py src/server.py src/tracing.py src/mcp_kb/
```

- [ ] **Step 2: Rewrite imports** `from src.` -> `from mcp_kb.` in `src/mcp_kb/*.py` and `tests/*.py`.

- [ ] **Step 3: Package config in `pyproject.toml`**

```toml
[project.scripts]
kb = "mcp_kb.cli:app"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Reinstall and run tests**

Run: `uv pip install -q --python .venv/bin/python -e ".[dev]" && .venv/bin/python -m pytest tests -q`
Expected: `10 passed`, with no `PYTHONPATH` needed.

- [ ] **Step 5: Commit** `refactor: package code as mcp_kb`

### Task 3: Citations use the path relative to the docs root

**Files:** modify `src/mcp_kb/indexer.py`; test `tests/test_indexer.py`

- [ ] **Step 1: Failing test**

```python
def test_sources_use_relative_paths(tmp_path):
    from mcp_kb.searcher import list_sources
    docs = tmp_path / "docs"
    (docs / "a").mkdir(parents=True)
    (docs / "b").mkdir()
    (docs / "a" / "README.md").write_text("# A\n\nalpha service runbook\n")
    (docs / "b" / "README.md").write_text("# B\n\nbeta service runbook\n")
    settings = Settings(docs_path=docs, db_path=tmp_path / "chroma")
    build_index(settings)
    assert list_sources(settings) == ["a/README.md", "b/README.md"]
```

- [ ] **Step 2: Run it** -> FAIL: `['README.md']`.

- [ ] **Step 3: Implement** in `build_index`:

```python
source = md_file.relative_to(settings.docs_path).as_posix()
metas.append({"source": source, "start_line": chunk["start_line"], "end_line": chunk["end_line"]})
```

- [ ] **Step 4: Run tests** -> all pass.

- [ ] **Step 5: Commit** `fix: cite paths relative to docs root so same-named files don't collide`

### Task 4: Empty index returns a clear message instead of crashing

**Files:** modify `src/mcp_kb/searcher.py`, `src/mcp_kb/config.py`; test `tests/test_searcher.py`

- [ ] **Step 1: Failing test**

```python
def test_empty_index_returns_nothing(tmp_path):
    settings = Settings(db_path=tmp_path / "empty")
    assert search("anything", settings) == []
    assert list_sources(settings) == []
```

- [ ] **Step 2: Run it** -> FAIL (`docs_path` required, then collection not found).

- [ ] **Step 3: Implement**

`config.py`: `docs_path: Path | None = None` (search and serve don't need it; `build_index` does).

`searcher.py`:
```python
COLLECTION = "kb_docs"


def _get_collection(settings: Settings):
    """Return the collection, or None if nothing has been indexed yet."""
    client = chromadb.PersistentClient(path=str(settings.db_path))
    if COLLECTION not in [c.name for c in client.list_collections()]:
        return None
    ef = SentenceTransformerEmbeddingFunction(model_name=settings.model_name)
    return client.get_collection(COLLECTION, embedding_function=ef)
```
`search` and `list_sources` return `[]` when `_get_collection` returns `None`.

- [ ] **Step 4: Run tests** -> all pass. **Step 5: Commit** `fix: handle a missing index gracefully`

### Task 5: Retrieval regression gate

**Files:** create `tests/test_retrieval_quality.py`

- [ ] **Step 1: Write the test** (the 20 questions from `eval.py`, unchanged)

```python
from pathlib import Path
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index
from mcp_kb.searcher import search

EXAMPLE_DOCS = Path(__file__).parent.parent / "example-docs"
GOLD = [  # (question, file that must be retrieved)
    ("how do I get AWS access on day one", "onboarding.md"),
    # ... all 20 pairs from eval.py ...
]


def test_example_docs_retrieval_does_not_regress(tmp_path):
    settings = Settings(docs_path=EXAMPLE_DOCS, db_path=tmp_path / "chroma")
    build_index(settings)
    reciprocal_ranks = []
    for question, expected in GOLD:
        sources = [r["source"] for r in search(question, settings)]
        reciprocal_ranks.append(1 / (sources.index(expected) + 1) if expected in sources else 0.0)
    recall = sum(rr > 0 for rr in reciprocal_ranks) / len(GOLD)
    mrr = sum(reciprocal_ranks) / len(GOLD)
    assert recall == 1.0, f"Recall@5 dropped to {recall:.3f}"
    assert mrr >= 0.90, f"MRR dropped to {mrr:.3f} (baseline 0.925)"
```

- [ ] **Step 2: Run** -> PASS (it locks in today's behavior). **Step 3: Commit** `test: add retrieval regression gate on example-docs`

### Task 6: `kb` CLI and an MCP server built from settings, warmed at start

**Files:** create `src/mcp_kb/cli.py`, `tests/test_server.py`; rewrite `src/mcp_kb/server.py`; add `warmup` to `searcher.py`; delete `index.py`, `eval.py`

- [ ] **Step 1: Failing end-to-end test**

```python
import sys
from pathlib import Path
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index

EXAMPLE_DOCS = Path(__file__).parent.parent / "example-docs"


async def test_search_docs_over_stdio(tmp_path):
    db = tmp_path / "chroma"
    build_index(Settings(docs_path=EXAMPLE_DOCS, db_path=db))
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_kb.cli", "serve", "--db", str(db)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"search_docs", "list_docs"}
            result = await session.call_tool("search_docs", {"query": "how do I roll back a deployment"})
            assert "[source: deployment-runbook.md:L" in result.content[0].text
```

- [ ] **Step 2: Run** -> FAIL (`No module named mcp_kb.cli`).

- [ ] **Step 3: `searcher.warmup`**

```python
def warmup(settings: Settings) -> None:
    """Load the embedding model now so the first real query isn't slow."""
    col = _get_collection(settings)
    if col is not None and col.count():
        col.query(query_texts=["warmup"], n_results=1)
```

- [ ] **Step 4: `server.py`** builds the server from `Settings` (closure over settings instead of module globals), keeps both tools and messages, and adds:

```python
async def run(settings: Settings) -> None:
    setup_tracing()
    warmup(settings)
    server = build_server(settings)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
```
`search_docs` on an empty index returns `No documents indexed. Run: kb index <path>`.

- [ ] **Step 5: `cli.py`**

```python
import asyncio
from pathlib import Path
from typing import Optional
import typer
from mcp_kb import searcher, server
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index

DEFAULT_DB = Path.home() / ".mcp-kb" / "chroma"
app = typer.Typer(no_args_is_help=True, help="Local semantic search over your markdown docs, for Claude.")


@app.command()
def index(docs: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of .md files"),
          db: Path = typer.Option(DEFAULT_DB, "--db", envvar="KB_DB_PATH")):
    """Index a folder of markdown docs."""
    count = build_index(Settings(docs_path=docs, db_path=db))
    typer.echo(f"Indexed {count} chunks from {docs}")
    typer.echo(f"Connect to Claude Code: claude mcp add kb -- kb serve --db {db}")


@app.command()
def search(query: str, db: Path = typer.Option(DEFAULT_DB, "--db", envvar="KB_DB_PATH"),
           top_k: int = typer.Option(5, "--top-k")):
    """Search the index from the terminal (same results Claude gets)."""
    results = searcher.search(query, Settings(db_path=db, top_k=top_k))
    if not results:
        typer.echo("No results. Did you run: kb index <path>?")
    for r in results:
        typer.echo(f"{r['citation']}\n{r['text']}\n")


@app.command()
def serve(db: Path = typer.Option(DEFAULT_DB, "--db", envvar="KB_DB_PATH"),
          docs: Optional[Path] = typer.Option(None, "--docs", envvar="KB_DOCS_PATH")):
    """Run the MCP server over stdio (Claude starts this for you)."""
    asyncio.run(server.run(Settings(docs_path=docs, db_path=db)))


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Delete** `index.py`, `eval.py`; drop unused imports in `tracing.py` (`phoenix as px`, `TracerProvider`, `SimpleSpanProcessor`).

- [ ] **Step 7: Run all tests** -> all pass. **Step 8: Commit** `feat: kb CLI, settings-based MCP server, model warmup at start`

### Task 7: README, CI, spec correction

- [ ] **Step 1: README:** quickstart becomes `uv venv && uv pip install -e .`, `kb index ./example-docs`, `kb search "how do I roll back"`; Claude Desktop config uses the absolute path to `.venv/bin/kb` with args `["serve", "--db", ...]`; badge and clone URLs point to `MCP_Knowledge_Base`; remove the MS MARCO section and "10 tests" line; add a short "Benchmarks: BEIR results coming in v2" note.
- [ ] **Step 2: CI:** cache `~/.cache/huggingface` with `actions/cache@v4` keyed on `pyproject.toml`.
- [ ] **Step 3: Spec:** replace "Model and DB loaded on every query / Seconds per call" with "Model loads on the first query (2.3 s cold start) / Warm at server start".
- [ ] **Step 4: Verify:** full test suite, `kb --help`, `kb index ./example-docs`, `kb search "roll back"`, and time two searches.
- [ ] **Step 5: Commit** `docs: README and CI for the kb command`

---

## Exit check (from spec)

- All tests pass, including the regression gate and the stdio test.
- First `search_docs` call after server start is under 500 ms (warmup moved the cold start to startup).
- `git ls-files` shows no egg-info, `.DS_Store`, old plans or MS MARCO script.

## Interview questions for this phase

1. What does `pip install -e .` do, and why does `[project.scripts]` give you a `kb` command?
2. Why was `from src.config import ...` a problem?
3. Why move model loading to startup instead of caching it per call?
4. What does the regression gate protect against?
