"""BM25 keyword search over chunks, using SQLite FTS5 (built into Python, no extra dependency)."""
import re
import sqlite3
from pathlib import Path

FILENAME = "bm25.sqlite"
WORD = re.compile(r"\w+")


def write(db_dir: Path, rows: list[dict]) -> None:
    """Replace the keyword index with these chunks: [{"id", "text", "source", "start_line", "end_line"}]."""
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / FILENAME
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        # porter = stemming ("restoring" matches "restore"); UNINDEXED columns are stored, not searched
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5(text, id UNINDEXED, source UNINDEXED, "
                    "start_line UNINDEXED, end_line UNINDEXED, tokenize='porter unicode61')")
        con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                        [(r["text"], r["id"], r["source"], r["start_line"], r["end_line"]) for r in rows])
    con.close()


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
