import asyncio
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from mcp_kb.config import Settings
from mcp_kb.searcher import search, list_sources
from mcp_kb.tracing import setup_tracing

DOCS_PATH = Path(os.environ.get("KB_DOCS_PATH", "./docs"))
DB_PATH = Path(os.environ.get("KB_DB_PATH", str(Path.home() / ".mcp-kb" / "chroma")))

settings = Settings(docs_path=DOCS_PATH, db_path=DB_PATH)
server = Server("mcp-knowledge-base")
setup_tracing()


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_docs",
            description="Search the indexed markdown knowledge base. Returns relevant chunks with source citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language question or keyword search"}
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_docs",
            description="List all markdown files currently indexed in the knowledge base.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_docs":
        results = search(arguments["query"], settings)
        if not results:
            return [types.TextContent(type="text", text="No results found.")]
        lines = [f"{r['citation']}\n{r['text']}\n" for r in results]
        return [types.TextContent(type="text", text="\n---\n".join(lines))]

    if name == "list_docs":
        sources = list_sources(settings)
        if not sources:
            return [types.TextContent(type="text", text="No documents indexed. Run: python3 index.py --docs <path>")]
        return [types.TextContent(type="text", text="Indexed documents:\n" + "\n".join(f"- {s}" for s in sources))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
