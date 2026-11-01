import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from mcp_kb import bm25
from mcp_kb.config import Settings
from mcp_kb.fusion import rrf
from mcp_kb.tracing import search_span

COLLECTION = "kb_docs"
MODES = ("dense", "bm25", "hybrid")


def _get_collection(settings: Settings):
    """Return the collection, or None if nothing has been indexed yet."""
    client = chromadb.PersistentClient(path=str(settings.db_path))
    if COLLECTION not in [c.name for c in client.list_collections()]:
        return None
    ef = SentenceTransformerEmbeddingFunction(model_name=settings.model_name)
    return client.get_collection(COLLECTION, embedding_function=ef)


def warmup(settings: Settings) -> None:
    """Load the embedding model now so the first real query isn't slow."""
    col = _get_collection(settings)
    if col is not None and col.count():
        col.query(query_texts=["warmup"], n_results=1)


def _dense(col, query: str, n: int) -> list[dict]:
    r = col.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
    return [{"id": i, "text": t, **m} for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]


def search(query: str, settings: Settings) -> list[dict]:
    """Top chunks for a query, using settings.search_mode: dense, bm25, or hybrid (RRF of both)."""
    if settings.search_mode not in MODES:
        raise ValueError(f"search_mode must be one of {MODES}, got {settings.search_mode!r}")
    col = _get_collection(settings)
    if col is None:
        return []
    with search_span(query) as span:
        if settings.search_mode == "dense":
            hits = _dense(col, query, settings.top_k)
        elif settings.search_mode == "bm25":
            hits = bm25.query(settings.db_path, query, settings.top_k)
        else:
            dense_hits = _dense(col, query, settings.candidates)
            keyword_hits = bm25.query(settings.db_path, query, settings.candidates)
            by_id = {h["id"]: h for h in keyword_hits + dense_hits}
            fused = rrf([[h["id"] for h in dense_hits], [h["id"] for h in keyword_hits]])
            hits = [by_id[i] for i in fused[: settings.top_k]]
        output = [{"text": h["text"], "source": h["source"],
                   "citation": f"[source: {h['source']}:L{h['start_line']}-L{h['end_line']}]"} for h in hits]
        span.record_results(output)
    return output


def list_sources(settings: Settings) -> list[str]:
    """Return sorted unique filenames in the index."""
    col = _get_collection(settings)
    if col is None:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in all_meta})
