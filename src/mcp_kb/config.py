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
