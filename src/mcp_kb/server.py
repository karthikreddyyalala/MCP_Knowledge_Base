"""MCP server: exposes search_docs and list_docs to Claude over stdio."""
from mcp.server.mcpserver import MCPServer

from mcp_kb.config import Settings
from mcp_kb.searcher import list_sources, search, warmup
from mcp_kb.tracing import setup_tracing

NOT_INDEXED = "No documents indexed. Run: kb index <path-to-your-docs>"
NOT_FOUND = "No relevant results in your docs."


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
        return "\n---\n".join(f"{r['citation']}\n{r['text']}\n" for r in results)

    @server.tool(structured_output=False)
    def list_docs() -> str:
        """List all markdown files currently indexed in the knowledge base."""
        sources = list_sources(settings)
        if not sources:
            return NOT_INDEXED
        return "Indexed documents:\n" + "\n".join(f"- {s}" for s in sources)

    return server


def run(settings: Settings) -> None:
    setup_tracing()
    warmup(settings)  # pay the model load at startup, not on the user's first question
    build_server(settings).run("stdio")
