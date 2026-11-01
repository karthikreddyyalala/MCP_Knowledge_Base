"""BEIR benchmark runner.

Indexes each BEIR document as `<doc_id>.md` through the real `build_index` and queries it
through the real `search`, so the numbers measure the product, not a separate script.
"""
import csv
import dataclasses
import datetime
import hashlib
import importlib.metadata as md
import json
import platform
import re
import statistics
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from mcp_kb.config import Settings
from mcp_kb.evals.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k
from mcp_kb.indexer import build_index
from mcp_kb.searcher import search

DATASETS = ("scifact", "nfcorpus", "fiqa")
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
K = 10
CANDIDATE_CHUNKS = 30  # chunks fetched per query; several can belong to the same doc
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# setup name -> (Settings overrides, search function). Later phases add hybrid / rerank here.
# Results for "v1-dense" (400-word chunks) were produced at commit 1016550 and are kept as history.
SETUPS: dict[str, tuple[dict, Callable[[str, Settings], list[dict]]]] = {
    "dense": ({}, search),
    "dense-no-prefix": ({"context_prefix": False}, search),
}
INDEX_FIELDS = ("model_name", "max_tokens", "overlap_tokens", "context_prefix")


def download(name: str, root: Path) -> Path:
    """Download and unzip a BEIR dataset once; return its folder."""
    root.mkdir(parents=True, exist_ok=True)
    folder = root / name
    if not (folder / "corpus.jsonl").exists():
        zip_path = root / f"{name}.zip"
        urllib.request.urlretrieve(URL.format(name), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        zip_path.unlink()
    return folder


def load(path: Path):
    """Return (corpus, queries, qrels) for the test split. Only judged queries are kept."""
    corpus = {}
    for line in (path / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        corpus[d["_id"]] = {"title": d.get("title", ""), "text": d["text"]}

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / "test.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)

    queries = {}
    for line in (path / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        if q["_id"] in qrels:
            queries[q["_id"]] = q["text"]
    return corpus, queries, qrels


def write_corpus(corpus: dict, out_dir: Path) -> None:
    """One markdown file per document, named by doc id, so every tool indexes identical files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for did, d in corpus.items():
        assert SAFE_ID.match(did), f"unsafe doc id: {did!r}"
        title = d["title"].strip() or did
        (out_dir / f"{did}.md").write_text(f"# {title}\n\n{d['text']}\n", encoding="utf-8")


def doc_ids_from_chunks(results: list[dict]) -> list[str]:
    """Collapse chunk hits to doc ids in rank order: a doc ranks where its best chunk ranks."""
    return list(dict.fromkeys(Path(r["source"]).stem for r in results))


def _ensure_index(corpus: dict, work: Path, overrides: dict) -> Settings:
    """Build one index per dataset and chunking config; setups that share a config share the index."""
    base = Settings(**overrides)
    config = {f: getattr(base, f) for f in INDEX_FIELDS} | {"n_docs": len(corpus)}
    key = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    settings = dataclasses.replace(base, docs_path=work / "docs", db_path=work / f"chroma-{key}",
                                   top_k=CANDIDATE_CHUNKS)
    marker = settings.db_path / "index.json"
    if marker.exists() and json.loads(marker.read_text()) == config:
        return settings
    if not any(settings.docs_path.glob("*.md")):
        write_corpus(corpus, settings.docs_path)
    build_index(settings)
    marker.write_text(json.dumps(config))
    return settings


def _environment(settings: Settings) -> dict:
    v = lambda pkg: md.version(pkg)  # noqa: E731
    return {
        "date": datetime.date.today().isoformat(),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "chromadb": v("chromadb"),
        "sentence-transformers": v("sentence-transformers"),
        "model": settings.model_name,
        **{f: getattr(settings, f) for f in INDEX_FIELDS if f != "model_name"},
    }


def run(name: str, corpus: dict, queries: dict, qrels: dict, work_dir: Path, setup: str) -> dict:
    overrides, search_fn = SETUPS[setup]
    settings = _ensure_index(corpus, work_dir / name, overrides)
    search_fn("warmup", settings)  # keep model load out of the latency numbers

    ranked_run, latencies = {}, []
    for qid, text in queries.items():
        t = time.perf_counter()
        hits = search_fn(text, settings)
        latencies.append((time.perf_counter() - t) * 1000)
        ranked_run[qid] = doc_ids_from_chunks(hits)[:K]

    per_q = lambda f: mean([f(ranked_run[q], qrels[q], K) for q in queries])  # noqa: E731
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0]
    return {
        "dataset": name,
        "setup": setup,
        "metrics": {
            "ndcg@10": round(per_q(ndcg_at_k), 4),
            "recall@10": round(per_q(recall_at_k), 4),
            "mrr@10": round(per_q(mrr_at_k), 4),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(p95, 1),
            "n_queries": len(queries),
            "n_docs": len(corpus),
        },
        "environment": _environment(settings),
        "run": ranked_run,
    }


def save(result: dict, results_dir: Path) -> Path:
    out = results_dir / result["dataset"] / f"{result['setup']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    write_summary(results_dir)
    return out


def write_summary(results_dir: Path) -> None:
    """Regenerate results/summary.md from every saved result file."""
    rows = ["| Dataset | Setup | nDCG@10 | Recall@10 | MRR@10 | p50 ms | p95 ms | Queries |",
            "|---|---|---|---|---|---|---|---|"]
    for f in sorted(results_dir.glob("*/*.json")):
        r = json.loads(f.read_text())
        m = r["metrics"]
        rows.append(f"| {r['dataset']} | {r['setup']} | {m['ndcg@10']:.3f} | {m['recall@10']:.3f} | "
                    f"{m['mrr@10']:.3f} | {m['p50_ms']} | {m['p95_ms']} | {m['n_queries']} |")
    (results_dir / "summary.md").write_text(
        "# Benchmark results\n\nGenerated by `kb bench`. Do not edit by hand.\n\n" + "\n".join(rows) + "\n")
