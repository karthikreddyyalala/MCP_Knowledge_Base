from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp_kb import bm25
from mcp_kb.chunking import chunk_markdown
from mcp_kb.config import Settings
from mcp_kb.models import get_tokenizer


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

    tokenizer = get_tokenizer(settings.model_name)
    docs, metas, ids = [], [], []
    chunk_id = 0

    for md_file in sorted(settings.docs_path.glob("**/*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Relative path, not just the filename: a/README.md and b/README.md must not collide.
        source = md_file.relative_to(settings.docs_path).as_posix()
        for chunk in chunk_markdown(text, source, tokenizer, settings.max_tokens,
                                    settings.overlap_tokens, settings.context_prefix):
            docs.append(chunk["text"])
            metas.append({"source": source, "start_line": chunk["start_line"], "end_line": chunk["end_line"]})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    bm25.write(settings.db_path, [{"id": i, "text": t, **m} for i, t, m in zip(ids, docs, metas)])

    # Chroma rejects a single add() above its max batch size, so large folders go in batches.
    batch = client.get_max_batch_size()
    for i in range(0, len(docs), batch):
        col.add(documents=docs[i:i + batch], metadatas=metas[i:i + batch], ids=ids[i:i + batch])

    return len(docs)
