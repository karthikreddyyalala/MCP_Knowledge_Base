"""`kb` command: index docs, search from the terminal, run the MCP server."""
from pathlib import Path
from typing import Optional

import typer

from mcp_kb import searcher, server
from mcp_kb.config import Settings, save_calibration, with_calibration
from mcp_kb.indexer import sync

DEFAULT_DB = Path.home() / ".mcp-kb" / "chroma"
app = typer.Typer(no_args_is_help=True, help="Local semantic search over your markdown docs, for Claude.")

DbOption = typer.Option(DEFAULT_DB, "--db", envvar="KB_DB_PATH", help="Where the index is stored")


@app.command()
def index(
    docs: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of .md files"),
    db: Path = DbOption,
    full: bool = typer.Option(False, "--full", help="Rebuild from scratch instead of syncing changes"),
):
    """Index a folder of .md/.txt docs. Re-running only re-indexes files that changed."""
    report = sync(Settings(docs_path=docs, db_path=db), full=full)
    typer.echo(("Full rebuild: " if report.full_rebuild else "Synced: ") + report.summary())
    typer.echo(f"Connect to Claude Code: claude mcp add kb -- kb serve --docs {docs.resolve()} --db {db}")


@app.command()
def search(query: str, db: Path = DbOption, top_k: int = typer.Option(5, "--top-k")):
    """Search from the terminal. Same results Claude gets."""
    settings = with_calibration(Settings(db_path=db, top_k=top_k))
    results = searcher.search(query, settings)
    if not results:
        typer.echo(server.NOT_FOUND if searcher.list_sources(settings) else server.NOT_INDEXED)
    for r in results:
        typer.echo(f"{r['citation']}\n{r['text']}\n")


@app.command()
def serve(
    db: Path = DbOption,
    docs: Optional[Path] = typer.Option(None, "--docs", envvar="KB_DOCS_PATH", help="Docs folder; changed files are re-indexed at startup"),
):
    """Run the MCP server over stdio. Claude starts this for you."""
    server.run(with_calibration(Settings(docs_path=docs, db_path=db)))


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


@app.command("eval")
def eval_(
    docs: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of .md/.txt docs"),
    db: Path = DbOption,
    n: int = typer.Option(50, "--n", help="Chunks to sample; each yields up to 2 questions"),
    model: str = typer.Option("llama3.2:3b", "--model", help="Ollama model that writes the questions"),
    gold: Optional[Path] = typer.Option(None, "--gold", help="Reuse a saved gold.jsonl instead of generating"),
    review: bool = typer.Option(False, "--review", help="Spot-check 10 questions by hand"),
    save_threshold: bool = typer.Option(False, "--save-threshold", help="Use the calibrated threshold from now on"),
    seed: int = typer.Option(0, "--seed"),
):
    """Measure search on YOUR docs: generate test questions locally, compare setups, calibrate "not found"."""
    import json
    import random

    from mcp_kb.evals import llm, own, synth

    settings = Settings(docs_path=docs, db_path=db)
    typer.echo("Syncing index: " + sync(settings).summary())
    if gold:
        items = synth.load_gold(gold)
        stats = None
    else:
        try:
            generate = llm.ollama(model=model, seed=seed)
        except llm.LLMUnavailable as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        typer.echo(f"Writing test questions with {model} (local)...")
        items, stats = synth.generate_gold(settings, generate, n=n, seed=seed, log=lambda m: typer.echo(m, err=True))
        gold = db / "eval" / "gold.jsonl"
        synth.save_gold(gold, items)
        typer.echo(f"Saved {len(items)} questions to {gold} (filters: {stats})")

    report = own.evaluate(settings, items)
    report["gold"], report["generation"], report["model"] = str(gold), stats, model
    if review:
        sample = random.Random(seed).sample(items, min(10, len(items)))
        ok = 0
        for g in sample:
            kind = f"answered by {g['source']}:L{g['start_line']}-L{g['end_line']}" if g["answerable"] else "NOT answered by the docs"
            ok += typer.confirm(f"\n{g['question']}\n  claimed: {kind}\n  fair test question?", default=True)
        report["review_pass_rate"] = ok / len(sample)
    typer.echo("\n" + own.format_report(report))

    out = db / "eval" / f"report-{report['date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    typer.echo(f"\nFull report: {out}")
    if save_threshold and "min_answered_0.97" in report["abstention"]:
        t = report["abstention"]["min_answered_0.97"]["threshold"]
        save_calibration(db, t, source=f"kb eval on {docs.resolve()} ({report['n_answerable']} + {report['n_unanswerable']} questions)")
        typer.echo(f"Saved not-found threshold {t} for this index; kb search and kb serve will use it.")


if __name__ == "__main__":
    app()
