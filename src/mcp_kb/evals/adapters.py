"""Run other local retrieval tools on BEIR, through their MCP servers, the way Claude would use them.

Every system indexes the same <doc_id>.md files written by beir.write_corpus, gets the same test
queries, and is scored by the same metrics code. Each tool runs with its documented defaults.
Models load once per server process, so per-query latency is comparable to `kb serve`.
"""
import asyncio
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@dataclass
class Adapter:
    name: str
    binary: Path

    def version(self) -> str:
        raise NotImplementedError

    def prepare(self, docs: Path, work: Path) -> None:
        """Index docs (once; reuse if already built)."""
        raise NotImplementedError

    def server(self, work: Path) -> StdioServerParameters:
        raise NotImplementedError

    async def wait_ready(self, session: ClientSession) -> None:
        pass

    async def search(self, session: ClientSession, query: str, k: int) -> list[str]:
        """Return file paths in rank order."""
        raise NotImplementedError


class Qmd(Adapter):
    """tobi/qmd: BM25 + vectors + LLM query expansion + LLM reranking (its recommended `query` mode)."""

    def _env(self, work: Path) -> dict:
        # keep qmd's models and index inside work, away from the user's ~/.cache/qmd
        return {**os.environ, "XDG_CACHE_HOME": str(work / "cache"), "XDG_CONFIG_HOME": str(work / "config")}

    def version(self) -> str:
        pkg = self.binary.resolve().parent.parent / "package.json"
        return json.loads(pkg.read_text())["version"] if pkg.exists() else "unknown"

    def prepare(self, docs: Path, work: Path) -> None:
        done = work / "indexed.ok"
        if done.exists():
            return
        work.mkdir(parents=True, exist_ok=True)
        run = lambda *a: subprocess.run([str(self.binary), "--index", "bench", *a], cwd=work,  # noqa: E731
                                        env=self._env(work), check=True, capture_output=True, text=True)
        run("collection", "add", str(docs), "--name", "beir")
        run("embed", "--timeout", "0")
        done.write_text("ok")

    def server(self, work: Path) -> StdioServerParameters:
        return StdioServerParameters(command=str(self.binary), args=["--index", "bench", "mcp"],
                                     env=self._env(work), cwd=str(work))

    async def search(self, session: ClientSession, query: str, k: int) -> list[str]:
        res = await session.call_tool("query", {"query": query, "limit": k})
        data = res.structured_content or json.loads(res.content[0].text)
        return [r["file"] for r in data.get("results", [])]


class KnowledgeRag(Adapter):
    """lyonzin/knowledge-rag: hybrid BM25 + vectors with cross-encoder reranking (`search_knowledge`)."""

    def _env(self, work: Path) -> dict:
        return {**os.environ, "KNOWLEDGE_RAG_DIR": str(work)}

    def version(self) -> str:
        site = next(self.binary.resolve().parent.parent.glob("lib/python*/site-packages"), None)
        dist = next(site.glob("knowledge_rag-*.dist-info"), None) if site else None
        return dist.name.split("-")[1] if dist else "unknown"

    def prepare(self, docs: Path, work: Path) -> None:
        work.mkdir(parents=True, exist_ok=True)
        if not (work / "documents").exists():
            shutil.copytree(docs, work / "documents")  # it indexes <dir>/documents on startup

    def server(self, work: Path) -> StdioServerParameters:
        return StdioServerParameters(command=str(self.binary), args=[], env=self._env(work), cwd=str(work))

    async def wait_ready(self, session: ClientSession) -> None:
        """It indexes in the background after start; wait until that finishes."""
        start = time.time()
        while True:
            status = json.loads((await session.call_tool("get_reindex_status", {})).content[0].text)
            r = status.get("reindex", {})
            if not r.get("active") and r.get("last_result") and time.time() - start > 3:
                return
            if time.time() - start > 6 * 3600:
                raise TimeoutError("knowledge-rag indexing did not finish in 6 hours")
            await asyncio.sleep(5)

    async def search(self, session: ClientSession, query: str, k: int) -> list[str]:
        res = await session.call_tool("search_knowledge", {"query": query, "max_results": k})
        data = json.loads(res.content[0].text)
        return [r["source"] for r in data.get("results", [])]


ADAPTERS = {"qmd": Qmd, "knowledge-rag": KnowledgeRag}


def doc_ids(paths: list[str], known: dict[str, str]) -> list[str]:
    """Map returned file paths to BEIR doc ids (by file stem, case-insensitive), deduped in rank order."""
    out = []
    for p in paths:
        did = known.get(Path(p).stem.lower())
        if did and did not in out:
            out.append(did)
    return out


async def run_queries(adapter: Adapter, work: Path, queries: dict[str, str], doc_id_map: dict[str, str],
                      k: int, log=lambda m: None) -> tuple[dict[str, list[str]], list[float]]:
    ranked, latencies = {}, []
    async with stdio_client(adapter.server(work)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            log(f"{adapter.name}: waiting for its index")
            await adapter.wait_ready(session)
            await adapter.search(session, "warmup", k)  # load models outside the timing
            for i, (qid, text) in enumerate(queries.items(), 1):
                t = time.perf_counter()
                paths = await adapter.search(session, text, k)
                latencies.append((time.perf_counter() - t) * 1000)
                ranked[qid] = doc_ids(paths, doc_id_map)[:k]
                if i % 50 == 0:
                    log(f"{adapter.name}: {i}/{len(queries)} queries")
    return ranked, latencies
