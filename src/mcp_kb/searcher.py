import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp_kb.config import Settings
from mcp_kb.tracing import search_span


COLLECTION = "kb_docs"


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


def search(query: str, settings: Settings) -> list[dict]:
    col = _get_collection(settings)
    if col is None:
        return []
    with search_span(query) as span:
        results = col.query(query_texts=[query], n_results=settings.top_k, include=["documents", "metadatas"])
        output = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            citation = f"[source: {meta['source']}:L{meta['start_line']}-L{meta['end_line']}]"
            output.append({"text": text, "source": meta["source"], "citation": citation})
        span.record_results(output)
    return output


def list_sources(settings: Settings) -> list[str]:
    """Return sorted unique filenames in the index."""
    col = _get_collection(settings)
    if col is None:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in all_meta})
