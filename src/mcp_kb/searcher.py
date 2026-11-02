import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from mcp_kb import bm25
from mcp_kb.config import Settings
from mcp_kb.expand import expand
from mcp_kb.fusion import rrf
from mcp_kb.indexer import COLLECTION
from mcp_kb.models import get_reranker
from mcp_kb.tracing import search_span

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
        if settings.rerank:
            get_reranker(settings.rerank_model).predict([("warmup", "warmup")])


def _dense(col, query: str, n: int) -> list[dict]:
    r = col.query(query_texts=[query], n_results=n, include=["documents", "metadatas"])
    return [{"id": i, "text": t, **m} for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]


def _retrieve(col, query: str, settings: Settings, n: int) -> list[dict]:
    """Ranked candidates. One list per retriever and query variant, fused with RRF when there are several."""
    extra = expand(query, settings) if settings.expand_queries else {}
    dense_queries = [query] + ([extra["hypothetical"]] if "hypothetical" in extra else [])
    keyword_queries = [query] + ([extra["keywords"]] if "keywords" in extra else [])

    lists = []
    single = settings.search_mode != "hybrid" and not extra
    m = n if single else max(n, settings.candidates)
    if settings.search_mode in ("dense", "hybrid"):
        lists += [_dense(col, q, m) for q in dense_queries]
    if settings.search_mode in ("bm25", "hybrid"):
        lists += [bm25.query(settings.db_path, q, m) for q in keyword_queries]
    if len(lists) == 1:
        return lists[0][:n]
    by_id = {}
    for hits in lists:
        for h in hits:
            by_id.setdefault(h["id"], h)
    return [by_id[i] for i in rrf([[h["id"] for h in hits] for hits in lists])[:n]]


def _rerank(query: str, hits: list[dict], settings: Settings) -> list[dict]:
    scores = get_reranker(settings.rerank_model).predict([(query, h["text"]) for h in hits])
    for h, score in zip(hits, scores):
        h["score"] = round(float(score), 3)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def search(query: str, settings: Settings) -> list[dict]:
    """Top chunks for a query.

    1. retrieve candidates with settings.search_mode (dense, bm25, or hybrid = RRF of both)
    2. optionally rerank them with a cross-encoder
    3. return [] ("not found") if the best reranked score is below settings.min_score
    """
    if settings.search_mode not in MODES:
        raise ValueError(f"search_mode must be one of {MODES}, got {settings.search_mode!r}")
    col = _get_collection(settings)
    if col is None:
        return []
    with search_span(query) as span:
        n = max(settings.top_k, settings.rerank_candidates) if settings.rerank else settings.top_k
        hits = _retrieve(col, query, settings, n)
        if settings.rerank and hits:
            # rerank the head; anything past rerank_candidates keeps its retrieval order
            k = settings.rerank_candidates
            hits = _rerank(query, hits[:k], settings) + hits[k:]
            if settings.min_score is not None and hits[0]["score"] < settings.min_score:
                hits = []
        output = [{"text": h["text"], "source": h["source"], "score": h.get("score"),
                   "start_line": h["start_line"], "end_line": h["end_line"],
                   "citation": f"[source: {h['source']}:L{h['start_line']}-L{h['end_line']}]"}
                  for h in hits[: settings.top_k]]
        span.record_results(output)
    return output


def list_sources(settings: Settings) -> list[str]:
    """Return sorted unique filenames in the index."""
    col = _get_collection(settings)
    if col is None:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in all_meta})
