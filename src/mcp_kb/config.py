from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    docs_path: Path
    db_path: Path = field(default_factory=lambda: Path.home() / ".mcp-kb" / "chroma")
    top_k: int = 5
    model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 400
    chunk_overlap: int = 50
