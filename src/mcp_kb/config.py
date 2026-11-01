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
    rerank: bool = True  # hybrid + rerank top 20 won on BEIR (see README)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20  # 20 tied or beat 50 on BEIR at about half the latency
    # Below this reranker score, search answers "not found". Calibrated in results/abstention.json:
    # answers 97% of answerable BEIR queries and refuses 70% of off-topic ones. None = never refuse.
    min_score: float | None = -6.57
