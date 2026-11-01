"""Index a docs folder into ChromaDB (dense) and SQLite FTS5 (BM25), re-indexing only what changed.

manifest.json (next to the index) records the chunking config and, per file, a SHA-256 of its
content and the ids of its chunks. A sync compares the folder with the manifest, deletes the
chunks of changed or removed files, adds chunks for new or changed files, then writes the
manifest last. Writes are idempotent (upsert, delete-before-insert), so if a sync is
interrupted the next one repairs it.
"""
import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from mcp_kb import bm25
from mcp_kb.chunking import chunk_markdown
from mcp_kb.config import Settings
from mcp_kb.models import get_tokenizer

COLLECTION = "kb_docs"
MANIFEST = "manifest.json"
EXTENSIONS = (".md", ".txt")
# Changing any of these changes every chunk or vector, so it forces a full rebuild.
CONFIG_FIELDS = ("model_name", "max_tokens", "overlap_tokens", "context_prefix")


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # unreadable (not UTF-8)
    unchanged: int = 0
    total_chunks: int = 0
    full_rebuild: bool = False

    def summary(self) -> str:
        parts = [f"{len(self.added)} added", f"{len(self.changed)} changed", f"{len(self.removed)} removed",
                 f"{self.unchanged} unchanged"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped (not UTF-8: {', '.join(self.skipped)})")
        return ", ".join(parts) + f". {self.total_chunks} chunks in the index."


def _config(settings: Settings) -> dict:
    return {f: getattr(settings, f) for f in CONFIG_FIELDS}


def _chunk_id(source: str, n: int) -> str:
    """Stable per file, so a file's old chunks can be deleted exactly."""
    return f"{hashlib.sha1(source.encode()).hexdigest()[:12]}-{n}"


def scan(docs_path: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return ({source: (sha256, text)}, skipped). source is the path relative to docs_path."""
    files, skipped = {}, []
    for path in sorted(p for p in docs_path.rglob("*") if p.suffix in EXTENSIONS and p.is_file()):
        # Relative path, not just the filename: a/README.md and b/README.md must not collide.
        source = path.relative_to(docs_path).as_posix()
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(source)
            continue
        files[source] = (hashlib.sha256(raw).hexdigest(), text)
    return files, skipped


def read_manifest(settings: Settings) -> dict | None:
    path = settings.db_path / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def pending_changes(settings: Settings) -> int:
    """How many files differ from the index (added, changed or removed), without changing anything."""
    manifest = read_manifest(settings)
    if manifest is None or settings.docs_path is None:
        return 0
    current, _ = scan(settings.docs_path)
    old = manifest["files"]
    changed = sum(1 for s, (sha, _) in current.items() if old.get(s, {}).get("sha256") != sha)
    return changed + sum(1 for s in old if s not in current)


def sync(settings: Settings, full: bool = False) -> SyncReport:
    """Bring the index in line with settings.docs_path. full=True rebuilds from scratch."""
    if settings.docs_path is None:
        raise ValueError("sync needs settings.docs_path")
    db = settings.db_path
    db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db))
    ef = SentenceTransformerEmbeddingFunction(model_name=settings.model_name)
    report = SyncReport()

    manifest = read_manifest(settings)
    has_collection = COLLECTION in [c.name for c in client.list_collections()]
    if full or manifest is None or manifest["config"] != _config(settings) or not has_collection:
        if has_collection:
            client.delete_collection(COLLECTION)
        bm25.write(db, [])
        manifest = {"config": _config(settings), "files": {}}
        report.full_rebuild = True
    col = client.get_or_create_collection(COLLECTION, embedding_function=ef)

    current, report.skipped = scan(settings.docs_path)
    indexed = manifest["files"]

    # 1. drop chunks of files that changed or disappeared
    stale = [s for s in indexed if s not in current or indexed[s]["sha256"] != current[s][0]]
    stale_ids = [i for s in stale for i in indexed[s]["chunk_ids"]]
    if stale_ids:
        col.delete(ids=stale_ids)
        bm25.delete(db, stale_ids)
    for s in stale:
        (report.changed if s in current else report.removed).append(s)
        del indexed[s]

    # 2. chunk new and changed files
    tokenizer = get_tokenizer(settings.model_name)
    rows = []
    for source, (sha, text) in current.items():
        if source in indexed:
            report.unchanged += 1
            continue
        if source not in report.changed:
            report.added.append(source)
        chunks = chunk_markdown(text, source, tokenizer, settings.max_tokens,
                                settings.overlap_tokens, settings.context_prefix)
        ids = [_chunk_id(source, n) for n in range(len(chunks))]
        rows += [{"id": i, "text": c["text"], "source": source, "start_line": c["start_line"],
                  "end_line": c["end_line"]} for i, c in zip(ids, chunks)]
        indexed[source] = {"sha256": sha, "chunk_ids": ids}

    # 3. write both indexes (Chroma rejects a single call above its max batch size)
    batch = client.get_max_batch_size()
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        col.upsert(ids=[r["id"] for r in b], documents=[r["text"] for r in b],
                   metadatas=[{"source": r["source"], "start_line": r["start_line"], "end_line": r["end_line"]}
                              for r in b])
    bm25.add(db, rows)

    # 4. manifest last: if anything above failed, the next sync redoes this work
    manifest["indexed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    (db / MANIFEST).write_text(json.dumps(manifest))
    report.total_chunks = col.count()
    return report


def build_index(settings: Settings) -> int:
    """Full rebuild. Returns the number of chunks."""
    return sync(settings, full=True).total_chunks
