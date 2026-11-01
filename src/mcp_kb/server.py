"""MCP server: exposes search_docs and list_docs to Claude over stdio.

stdout carries MCP messages, so anything for humans goes to stderr.
"""
import sys

from mcp.server.mcpserver import MCPServer

from mcp_kb.config import Settings
from mcp_kb.indexer import pending_changes, read_manifest, sync
from mcp_kb.searcher import list_sources, search, warmup
from mcp_kb.tracing import setup_tracing

NOT_INDEXED = "No documents indexed. Run: kb index <path-to-your-docs>"
NOT_FOUND = "No relevant results in your docs."
LOW_CONFIDENCE = ("Low confidence: nothing scored as clearly relevant (best reranker score {score:.1f}). "
                  "These may not answer the question; say so if they don't.")


def format_results(results: list[dict], settings: Settings) -> str:
    """What Claude (and kb search) sees: citations and text, with a warning if every match is weak."""
    body = "\n---\n".join(f"{r['citation']}\n{r['text']}\n" for r in results)
    best = results[0].get("score")
    if settings.warn_below is not None and best is not None and best < settings.warn_below:
        return LOW_CONFIDENCE.format(score=best) + "\n\n" + body
    return body


def build_server(settings: Settings) -> MCPServer:
    server = MCPServer("mcp-knowledge-base")

    @server.tool(structured_output=False)
    def search_docs(query: str) -> str:
        """Search the indexed markdown knowledge base. Returns relevant chunks with source citations.

        Args:
            query: Natural language question or keyword search
        """
        results = search(query, settings)
        if not results:
            return NOT_INDEXED if not list_sources(settings) else NOT_FOUND
        return format_results(results, settings)

    @server.tool(structured_output=False)
    def list_docs() -> str:
        """List all markdown files currently indexed in the knowledge base."""
        sources = list_sources(settings)
        if not sources:
            return NOT_INDEXED
        manifest = read_manifest(settings) or {}
        lines = [f"Indexed documents ({len(sources)} files, last indexed {manifest.get('indexed_at', 'unknown')}):"]
        lines += [f"- {s}" for s in sources]
        pending = pending_changes(settings)
        if pending:
            lines.append(f"\n{pending} file(s) changed since then. Restart the server or run: kb index <path>")
        return "\n".join(lines)

    return server


def run(settings: Settings) -> None:
    setup_tracing()
    if settings.docs_path is not None:
        print(f"kb: syncing {settings.docs_path}: {sync(settings).summary()}", file=sys.stderr)
    warmup(settings)  # pay the model load at startup, not on the user's first question
    build_server(settings).run("stdio")
