"""Does kb eval tell the truth? Check it against human-labelled BEIR data.

kb eval ranks search setups using LLM-written questions. On a BEIR dataset we also know how the
same setups rank on human-written questions (results/<dataset>/*.json). If the two rankings agree,
kb eval is trustworthy on docs that have no human labels.
"""
import dataclasses
import datetime
import json
from pathlib import Path

from mcp_kb.evals import own
from mcp_kb.evals.beir import _ensure_index, download, load
from mcp_kb.llm import Generate
from mcp_kb.evals.synth import PROMPT_VERSION, generate_gold, overlap


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation: 1 = same order, 0 = unrelated, -1 = reversed."""
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def validate(dataset: str, work_dir: Path, results_dir: Path, generate: Generate, model: str,
             n: int = 100, seed: int = 0, log=lambda m: None, prompt_version: int = PROMPT_VERSION) -> dict:
    corpus, _, _ = load(download(dataset, work_dir / "datasets"))
    settings = _ensure_index(corpus, work_dir / "indexes" / dataset, {})
    gold, stats = generate_gold(settings, generate, n=n, seed=seed, log=log, prompt_version=prompt_version)
    report = own.evaluate(settings, gold)

    human = {}
    for setup in own.SETUPS:
        f = results_dir / dataset / f"{setup}.json"
        if f.exists():
            human[setup] = json.loads(f.read_text())["metrics"]["ndcg@10"]
    synthetic = {s: report["setups"][s]["mrr@5"] for s in human}
    answerable = [g for g in gold if g["answerable"]]
    chunk_text = {}  # mean word overlap of kept questions with their chunk: higher = more keyword-friendly
    for g in answerable:
        chunk_text[g["question"]] = (settings.docs_path / g["source"]).read_text()
    mean_overlap = (sum(overlap(q, t) for q, t in chunk_text.items()) / len(chunk_text)) if chunk_text else None

    return {
        "dataset": dataset, "model": model, "prompt_version": prompt_version, "n_chunks": n, "seed": seed,
        "date": datetime.date.today().isoformat(),
        "n_answerable": report["n_answerable"], "n_unanswerable": report["n_unanswerable"],
        "generation": stats, "mean_overlap_kept": round(mean_overlap, 3) if mean_overlap is not None else None,
        "human_ndcg@10": human, "synthetic_mrr@5": synthetic,
        "spearman": round(spearman(list(human.values()), list(synthetic.values())), 3),
        "best_human": max(human, key=human.get), "best_synthetic": max(synthetic, key=synthetic.get),
        "abstention": report["abstention"],
        "setups": {k: {m: v for m, v in d.items() if m != "per_question"} for k, d in report["setups"].items()},
    }
