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
import random
import re
import statistics
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from mcp_kb.config import Settings
from mcp_kb.bm25 import FILENAME as BM25_FILE
from mcp_kb.evals.abstain import calibrate
from mcp_kb.evals.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k
from mcp_kb.evals.stats import paired_randomization
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
    "bm25": ({"search_mode": "bm25"}, search),
    "hybrid": ({"search_mode": "hybrid"}, search),
    "dense-rerank": ({"search_mode": "dense", "rerank": True}, search),
    "hybrid-rerank": ({"search_mode": "hybrid", "rerank": True}, search),
    "hybrid-rerank20": ({"search_mode": "hybrid", "rerank": True, "rerank_candidates": 20}, search),
}
BASELINE_SETUP = "dense"  # summary reports delta and p-value against this row
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
    fresh = marker.exists() and json.loads(marker.read_text()) == config
    if fresh and (settings.db_path / BM25_FILE).exists():  # indexes built before phase 3 lack BM25
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


def _result(name: str, setup: str, corpus: dict, queries: dict, qrels: dict,
            ranked_run: dict, latencies: list[float], environment: dict) -> dict:
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
        "environment": environment,
        "per_query": {q: round(ndcg_at_k(ranked_run[q], qrels[q], K), 4) for q in queries},
        "run": ranked_run,
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
    return _result(name, setup, corpus, queries, qrels, ranked_run, latencies, _environment(settings))


def run_external(name: str, corpus: dict, queries: dict, qrels: dict, work_dir: Path,
                 system: str, binary: Path, log=lambda m: None) -> dict:
    """Benchmark another tool (see evals/adapters.py) on the same files, queries and metrics."""
    import asyncio

    from mcp_kb.evals.adapters import ADAPTERS, run_queries

    adapter = ADAPTERS[system](system, binary)
    docs = work_dir / name / "docs"
    if not any(docs.glob("*.md")):
        write_corpus(corpus, docs)
    work = work_dir / name / system
    t = time.perf_counter()
    log(f"{system}: indexing {len(corpus):,} docs (first run only)")
    adapter.prepare(docs, work)
    index_s = time.perf_counter() - t
    id_map = {did.lower(): did for did in corpus}
    ranked_run, latencies = asyncio.run(run_queries(adapter, work, queries, id_map, K, log))
    env = {"date": datetime.date.today().isoformat(), "python": platform.python_version(),
           "machine": f"{platform.system()} {platform.machine()}", "system": system,
           "version": adapter.version(), "settings": "tool defaults", "prepare_seconds": round(index_s, 1)}
    return _result(name, system, corpus, queries, qrels, ranked_run, latencies, env)


def save(result: dict, results_dir: Path) -> Path:
    out = results_dir / result["dataset"] / f"{result['setup']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    write_summary(results_dir)
    return out


def write_summary(results_dir: Path) -> None:
    """Regenerate results/summary.md from every saved result file.

    Each row shows its nDCG@10 change vs the baseline setup on the same dataset, with a paired
    randomization p-value over per-query nDCG@10. p >= 0.05 means the change may be noise.
    """
    rows = ["| Dataset | Setup | nDCG@10 | Δ vs dense | p | Recall@10 | MRR@10 | p50 ms | p95 ms | Queries |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        loaded = {f.stem: json.loads(f.read_text()) for f in sorted(ds_dir.glob("*.json"))}
        results = {k: v for k, v in loaded.items() if "metrics" in v}  # skip kb-eval validation files
        base = results.get(BASELINE_SETUP)
        for setup, r in results.items():
            m = r["metrics"]
            delta = p = "n/a"
            if base and setup != BASELINE_SETUP and "per_query" in r and "per_query" in base:
                qs = sorted(base["per_query"])
                a, b = [base["per_query"][q] for q in qs], [r["per_query"][q] for q in qs]
                delta = f"{m['ndcg@10'] - base['metrics']['ndcg@10']:+.3f}"
                p = f"{paired_randomization(a, b):.3f}"
            elif setup == BASELINE_SETUP:
                delta = p = "baseline"
            rows.append(f"| {r['dataset']} | {setup} | {m['ndcg@10']:.3f} | {delta} | {p} | {m['recall@10']:.3f} | "
                        f"{m['mrr@10']:.3f} | {m['p50_ms']} | {m['p95_ms']} | {m['n_queries']} |")
    (results_dir / "summary.md").write_text(
        "# Benchmark results\n\nGenerated by `kb bench`. Do not edit by hand.\n\n"
        "Δ and p compare nDCG@10 with the `dense` row of the same dataset (paired randomization test, "
        "10,000 permutations). p >= 0.05 means the difference may be noise.\n\n" + "\n".join(rows) + "\n")


def calibrate_abstention(work_dir: Path, setup: str = "hybrid-rerank20", per_side: int = 300, seed: int = 0) -> dict:
    """Pick the "not found" threshold using cross-dataset negatives.

    BEIR has no unanswerable questions, so for each dataset: its own queries are answerable
    and a sample of the other datasets' queries, run against the same index, are not
    (finance questions against a science corpus). Off-topic negatives are easier than
    near-miss ones, so the result is a starting default; `kb eval` recalibrates on your docs.
    """
    overrides, search_fn = SETUPS[setup]
    rng = random.Random(seed)
    data = {d: load(download(d, work_dir / "datasets")) for d in DATASETS}
    pos, neg, per_dataset = [], [], {}
    for name, (corpus, queries, _) in data.items():
        settings = dataclasses.replace(_ensure_index(corpus, work_dir / "indexes" / name, overrides), top_k=1)
        own = rng.sample(sorted(queries.values()), min(per_side, len(queries)))
        others = [q for other, (_, qs, _) in data.items() if other != name for q in qs.values()]
        foreign = rng.sample(sorted(others), min(per_side, len(others)))
        top = lambda q: (search_fn(q, settings) or [{"score": float("-inf")}])[0]["score"]  # noqa: E731
        p, n = [top(q) for q in own], [top(q) for q in foreign]
        pos += p
        neg += n
        per_dataset[name] = {"answerable": len(p), "unanswerable": len(n)}
    r3 = lambda d: {k: round(v, 3) for k, v in d.items()}  # noqa: E731
    return {"setup": setup, "method": "cross-dataset negatives",
            "balanced": r3(calibrate(pos, neg)),
            "min_answered_0.97": r3(calibrate(pos, neg, min_answered=0.97)),
            "n_answerable": len(pos), "n_unanswerable": len(neg), "per_dataset": per_dataset,
            "date": datetime.date.today().isoformat(),
            "scores": {"answerable": [round(x, 3) for x in pos], "unanswerable": [round(x, 3) for x in neg]}}
