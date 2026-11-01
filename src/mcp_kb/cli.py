"""`kb` command: index docs, search from the terminal, run the MCP server."""
from pathlib import Path
from typing import Optional

import typer

from mcp_kb import searcher, server
from mcp_kb.config import Settings
from mcp_kb.indexer import build_index

DEFAULT_DB = Path.home() / ".mcp-kb" / "chroma"
app = typer.Typer(no_args_is_help=True, help="Local semantic search over your markdown docs, for Claude.")

DbOption = typer.Option(DEFAULT_DB, "--db", envvar="KB_DB_PATH", help="Where the index is stored")


@app.command()
def index(
    docs: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of .md files"),
    db: Path = DbOption,
):
    """Index a folder of markdown docs."""
    count = build_index(Settings(docs_path=docs, db_path=db))
    typer.echo(f"Indexed {count} chunks from {docs}")
    typer.echo(f"Connect to Claude Code: claude mcp add kb -- kb serve --db {db}")


@app.command()
def search(query: str, db: Path = DbOption, top_k: int = typer.Option(5, "--top-k")):
    """Search from the terminal. Same results Claude gets."""
    results = searcher.search(query, Settings(db_path=db, top_k=top_k))
    if not results:
        typer.echo("No results. Did you run: kb index <path>?")
    for r in results:
        typer.echo(f"{r['citation']}\n{r['text']}\n")


@app.command()
def serve(
    db: Path = DbOption,
    docs: Optional[Path] = typer.Option(None, "--docs", envvar="KB_DOCS_PATH", help="Docs folder (used for re-sync in v2)"),
):
    """Run the MCP server over stdio. Claude starts this for you."""
    server.run(Settings(docs_path=docs, db_path=db))


@app.command()
def bench(
    dataset: str = typer.Option(..., "--dataset", help="scifact, nfcorpus or fiqa"),
    setup: str = typer.Option("dense", "--setup", help="Search setup to measure"),
    work_dir: Path = typer.Option(Path.home() / ".mcp-kb" / "bench", "--work-dir", help="Downloads and indexes"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", help="Where result JSON is written"),
):
    """Benchmark a search setup on a BEIR dataset. Writes results/<dataset>/<setup>.json."""
    from mcp_kb.evals import beir  # imported here so normal commands stay fast

    if dataset not in beir.DATASETS:
        raise typer.BadParameter(f"choose one of {', '.join(beir.DATASETS)}", param_hint="--dataset")
    if setup not in beir.SETUPS:
        raise typer.BadParameter(f"choose one of {', '.join(beir.SETUPS)}", param_hint="--setup")

    typer.echo(f"Loading {dataset}...")
    corpus, queries, qrels = beir.load(beir.download(dataset, work_dir / "datasets"))
    typer.echo(f"{len(corpus):,} docs, {len(queries):,} queries. Indexing (first run only) and searching...")
    result = beir.run(dataset, corpus, queries, qrels, work_dir / "indexes", setup)
    out = beir.save(result, results_dir)
    m = result["metrics"]
    typer.echo(f"{dataset} / {setup}: nDCG@10 {m['ndcg@10']:.3f}  Recall@10 {m['recall@10']:.3f}  "
               f"MRR@10 {m['mrr@10']:.3f}  p50 {m['p50_ms']} ms")
    typer.echo(f"Saved {out}")


@app.command()
def calibrate(
    work_dir: Path = typer.Option(Path.home() / ".mcp-kb" / "bench", "--work-dir"),
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    setup: str = typer.Option("hybrid-rerank20", "--setup"),
):
    """Pick the "not found" threshold on BEIR. Writes results/abstention.json."""
    import json

    from mcp_kb.evals import beir

    r = beir.calibrate_abstention(work_dir, setup)
    (results_dir / "abstention.json").write_text(json.dumps(r, indent=1))
    for name in ("balanced", "min_answered_0.97"):
        c = r[name]
        typer.echo(f"{name:18} threshold {c['threshold']:7.3f}  answered {c['answered_rate']:.1%}  "
                   f"refused off-topic {c['abstained_rate']:.1%}")


if __name__ == "__main__":
    app()
