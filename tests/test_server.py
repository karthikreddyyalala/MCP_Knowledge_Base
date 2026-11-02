"""End to end: start `kb serve` as a subprocess and talk MCP to it over stdio, like Claude does."""
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp_kb.config import Settings
from mcp_kb.indexer import build_index

EXAMPLE_DOCS = Path(__file__).parent.parent / "example-docs"


def _params(db: Path) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_kb.cli", "serve", "--db", str(db)])


async def test_search_docs_over_stdio(tmp_path):
    db = tmp_path / "chroma"
    build_index(Settings(docs_path=EXAMPLE_DOCS, db_path=db))
    async with stdio_client(_params(db)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"search_docs", "list_docs"}
            result = await session.call_tool("search_docs", {"query": "how do I roll back a deployment"})
            assert "[source: deployment-runbook.md:L" in result.content[0].text


async def test_empty_index_tells_user_what_to_run(tmp_path):
    async with stdio_client(_params(tmp_path / "empty")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_docs", {"query": "anything"})
            assert "kb index" in result.content[0].text


async def test_off_topic_question_is_flagged_low_confidence(tmp_path):
    db = tmp_path / "chroma"
    build_index(Settings(docs_path=EXAMPLE_DOCS, db_path=db))
    async with stdio_client(_params(db)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_docs", {"query": "what is the best pizza in Naples"})
            assert result.content[0].text.startswith("Low confidence:")


async def test_server_with_docs_picks_up_edits_at_startup(tmp_path):
    docs, db = tmp_path / "docs", tmp_path / "chroma"
    docs.mkdir()
    (docs / "a.md").write_text("# Deploys\n\nRun kubectl apply to ship a release.\n")
    build_index(Settings(docs_path=docs, db_path=db))
    (docs / "a.md").write_text("# Deploys\n\nShip releases with the zebracorn pipeline.\n")  # edit after indexing
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "mcp_kb.cli", "serve", "--db", str(db), "--docs", str(docs)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_docs", {"query": "how do I ship a release with zebracorn"})
            assert "zebracorn" in result.content[0].text
            listing = await session.call_tool("list_docs", {})
            assert "a.md" in listing.content[0].text and "changed since" not in listing.content[0].text
