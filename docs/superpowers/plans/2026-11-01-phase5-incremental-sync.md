# Phase 5: Incremental Sync

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `kb index` and server start re-index only files whose content changed; deleted files disappear from results; edits show up after restarting Claude.

**Architecture:** `indexer.sync(settings, full=False) -> SyncReport`. A `manifest.json` in the index folder records the chunking config and, per file, its SHA-256 and chunk ids. Sync scans `.md` and `.txt` files, deletes chunks of changed/removed files from ChromaDB and BM25, chunks and upserts new/changed files, then writes the manifest last. Chunk ids are `<sha1(path)[:12]>-<n>`, stable per file, so deletes are exact. A config change (model, token budget, prefix) or a missing manifest forces a full rebuild. Writes are idempotent (Chroma `upsert`; BM25 deletes ids before inserting), so a crash mid-sync is repaired by the next sync. `build_index` = `sync(full=True)`. `kb serve --docs` syncs at start, logging to stderr (stdout is the MCP channel). `list_docs` reports file count, chunk count, last indexed time and pending changes.

## Tests (`tests/test_sync.py`)
1. First sync adds all files; second sync reports everything unchanged and embeds nothing.
2. Editing a file: reported as changed; new text is findable (dense and BM25), old text is gone.
3. Deleting a file: reported as removed; gone from `list_sources` and search.
4. Rewriting a file with identical content: unchanged (hash, not mtime).
5. A non-UTF-8 file is skipped and reported; other files index.
6. Changing `max_tokens` forces a full rebuild.
7. `.txt` files are indexed.
8. Server started with `--docs` picks up an edit made after the last `kb index`.

## Interview questions
1. Why hash content instead of using modification time?
2. Why write the manifest last, and why upsert?
3. Why does a config change force a full rebuild?
4. Why must the server log to stderr?
