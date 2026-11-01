from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    docs_path: Path | None = None  # only needed for indexing
    db_path: Path = field(default_factory=lambda: Path.home() / ".mcp-kb" / "chroma")
    top_k: int = 5
    model_name: str = "all-MiniLM-L6-v2"
    max_tokens: int = 256  # all-MiniLM-L6-v2 reads at most 256 tokens; the rest would be dropped
    overlap_tokens: int = 32
    context_prefix: bool = True  # prepend "path > H1 > H2" to each chunk
    search_mode: str = "hybrid"  # "dense", "bm25" or "hybrid"; hybrid won on BEIR (see README)
    candidates: int = 50  # per-retriever candidates before fusion
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 50  # how many fused candidates the cross-encoder scores
    min_score: float | None = None  # reranker score below which search answers "not found"
