"""BM25 keyword search over chunks, using SQLite FTS5 (built into Python, no extra dependency)."""
import re
import sqlite3
from pathlib import Path

FILENAME = "bm25.sqlite"
WORD = re.compile(r"\w+")


# porter = stemming ("restoring" matches "restore"); UNINDEXED columns are stored, not searched
SCHEMA = ("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(text, id UNINDEXED, source UNINDEXED, "
          "start_line UNINDEXED, end_line UNINDEXED, tokenize='porter unicode61')")


def _connect(db_dir: Path) -> sqlite3.Connection:
    db_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_dir / FILENAME)
    con.execute(SCHEMA)
    return con


def delete(db_dir: Path, ids: list[str]) -> None:
    """Remove chunks by id."""
    con = _connect(db_dir)
    with con:
        for i in range(0, len(ids), 500):  # stay under SQLite's bound-parameter limit
            batch = ids[i:i + 500]
            con.execute(f"DELETE FROM chunks WHERE id IN ({','.join('?' * len(batch))})", batch)
    con.close()


def add(db_dir: Path, rows: list[dict]) -> None:
    """Insert chunks: [{"id", "text", "source", "start_line", "end_line"}]. Re-adding an id replaces it."""
    delete(db_dir, [r["id"] for r in rows])  # idempotent, so a sync interrupted halfway can simply rerun
    con = _connect(db_dir)
    with con:
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                        [(r["text"], r["id"], r["source"], r["start_line"], r["end_line"]) for r in rows])
    con.close()


def write(db_dir: Path, rows: list[dict]) -> None:
    """Replace the whole keyword index with these chunks."""
    (db_dir / FILENAME).unlink(missing_ok=True)
    add(db_dir, rows)


def query(db_dir: Path, text: str, n: int) -> list[dict]:
    """Top n chunks by BM25. Returns [] if there is no index or the query has no words."""
    path = db_dir / FILENAME
    words = WORD.findall(text)
    if not path.exists() or not words:
        return []
    # Quote every word so punctuation and FTS5 operators (AND, OR, NOT, *, quotes) are literal,
    # and OR them so any matching word counts, like classic BM25 retrieval.
    match = " OR ".join(f'"{w}"' for w in words)
    con = sqlite3.connect(path)
    try:
        rows = con.execute("SELECT id, text, source, start_line, end_line FROM chunks "
                           "WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?", (match, n)).fetchall()
    finally:
        con.close()
    return [{"id": i, "text": t, "source": s, "start_line": int(a), "end_line": int(b)} for i, t, s, a, b in rows]
