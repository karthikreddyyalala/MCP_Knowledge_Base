"""Optional LLM query expansion (off by default; Settings.expand_queries).

A local LLM rewrites the question into extra search keywords (for BM25) and a short made-up
passage that would answer it (for embeddings; the "HyDE" technique: an answer-shaped text often
sits closer to the real answer than the question does). Retrieval fuses these extra lists with
the originals. If the LLM is unavailable or replies badly, search simply runs without expansion.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

from mcp_kb import llm
from mcp_kb.config import Settings

PROMPT = '''Rewrite this search query to help find matching passages in a collection of documents.

Query: {query}

Reply with JSON only:
{{"keywords": "5 to 12 search keywords, including synonyms", "hypothetical": "one or two sentences that would answer the query, written like a passage from the documents"}}'''
FIELDS = ("keywords", "hypothetical")


def _generator(model: str) -> llm.Generate:
    return llm.ollama(model=model)


@lru_cache(maxsize=None)
def _cached_generator(model: str) -> llm.Generate | None:
    try:
        return _generator(model)
    except llm.LLMUnavailable as e:
        print(f"kb: query expansion off ({e.args[0].splitlines()[0]})", file=sys.stderr)
        return None


class _DiskCache:
    """Expansions keyed by model and query, appended to a JSONL file next to the index."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.data[row["key"]] = row["value"]

    def add(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")


@lru_cache(maxsize=None)
def _cache(path: Path) -> _DiskCache:
    return _DiskCache(path)


def expand(query: str, settings: Settings) -> dict:
    """{"keywords": ..., "hypothetical": ...}, or {} if expansion isn't possible."""
    cache = _cache(settings.db_path / "expansions.jsonl") if settings.expand_cache else None
    key = f"{settings.expand_model}\t{query}"
    if cache is not None and key in cache.data:
        return cache.data[key]
    generate = _cached_generator(settings.expand_model)
    if generate is None:
        return {}
    try:
        reply = generate(PROMPT.format(query=query))
    except Exception as e:  # a flaky local LLM must never break search
        print(f"kb: query expansion failed for one query ({e})", file=sys.stderr)
        return {}
    out = {k: str(reply.get(k, "")).strip() for k in FIELDS if str(reply.get(k, "")).strip()}
    if cache is not None and out:
        cache.add(key, out)
    return out
