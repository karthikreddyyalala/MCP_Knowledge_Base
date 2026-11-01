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


if __name__ == "__main__":
    app()
