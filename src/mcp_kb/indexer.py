from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp_kb.config import Settings


def chunk_markdown(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[dict]:
    """Split markdown into chunks; each chunk is prefixed with its nearest ## header."""
    import re
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    # Split into sections at ## headers
    sections: list[tuple[str, list[tuple[int, str]]]] = []
    current_header = ""
    current_lines: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if re.match(r"^#{1,3} ", line):
            if current_lines:
                sections.append((current_header, current_lines))
            current_header = line.rstrip()
            current_lines = []
        else:
            current_lines.append((i, line))
    if current_lines:
        sections.append((current_header, current_lines))

    chunks = []
    for header, line_tuples in sections:
        body_words: list[str] = []
        line_numbers: list[int] = []
        for lineno, line in line_tuples:
            for word in line.split():
                body_words.append(word)
                line_numbers.append(lineno)

        if not body_words:
            continue

        start = 0
        while start < len(body_words):
            end = min(start + chunk_size, len(body_words))
            chunk_text = " ".join(body_words[start:end])
            if header:
                chunk_text = header + "\n\n" + chunk_text
            chunks.append({
                "text": chunk_text,
                "start_line": line_numbers[start],
                "end_line": line_numbers[end - 1],
            })
            if end == len(body_words):
                break
            start = end - chunk_overlap

    return chunks


def build_index(settings: Settings) -> int:
    """Index all .md files under settings.docs_path. Returns total chunk count."""
    settings.db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.db_path))
    ef = SentenceTransformerEmbeddingFunction(model_name=settings.model_name)

    try:
        client.delete_collection("kb_docs")
    except Exception:
        pass
    col = client.create_collection("kb_docs", embedding_function=ef)

    docs, metas, ids = [], [], []
    chunk_id = 0

    for md_file in sorted(settings.docs_path.glob("**/*.md")):
        text = md_file.read_text(encoding="utf-8")
        for chunk in chunk_markdown(text, settings.chunk_size, settings.chunk_overlap):
            docs.append(chunk["text"])
            metas.append({"source": md_file.name, "start_line": chunk["start_line"], "end_line": chunk["end_line"]})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    if docs:
        col.add(documents=docs, metadatas=metas, ids=ids)

    return len(docs)
