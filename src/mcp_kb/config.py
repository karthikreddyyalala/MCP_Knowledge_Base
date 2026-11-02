import datetime
import json
from dataclasses import dataclass, field, replace
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
    # Below this reranker score, results are returned with a low-confidence warning. Calibrated on
    # BEIR (results/abstention.json: 97% of answerable queries score above it). It is a warning, not
    # a refusal, because on short engineering docs correct answers can score below it.
    warn_below: float | None = -6.57
    # Hard refusal ("not found") below this score. Off by default; `kb eval --save-threshold`
    # sets it from your own docs (see config.with_calibration).
    min_score: float | None = None
    # Optional LLM query expansion through Ollama (see expand.py). Off by default: it adds an LLM
    # call to every search.
    expand_queries: bool = False
    expand_model: str = "llama3.2:3b"
    expand_cache: bool = True  # remember expansions next to the index


CALIBRATION_FILE = "calibration.json"


def save_calibration(db_path: Path, min_score: float, source: str) -> None:
    """Store a "not found" threshold next to the index (written by kb eval --save-threshold)."""
    db_path.mkdir(parents=True, exist_ok=True)
    (db_path / CALIBRATION_FILE).write_text(json.dumps(
        {"min_score": min_score, "source": source, "date": datetime.date.today().isoformat()}))


def with_calibration(settings: Settings) -> Settings:
    """Use the threshold calibrated on this index's own docs, if kb eval saved one."""
    path = settings.db_path / CALIBRATION_FILE
    if not path.exists():
        return settings
    return replace(settings, min_score=json.loads(path.read_text())["min_score"])
