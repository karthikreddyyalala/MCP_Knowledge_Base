"""Evaluate search on the user's own gold set and recommend a setup and threshold."""
import dataclasses
import datetime
import statistics
import time

from mcp_kb.config import Settings
from mcp_kb.evals.abstain import calibrate
from mcp_kb.evals.metrics import mean
from mcp_kb.evals.stats import paired_randomization
from mcp_kb.searcher import search

# ranking only: min_score is switched off while measuring them
SETUPS = {
    "dense": {"search_mode": "dense", "rerank": False},
    "bm25": {"search_mode": "bm25", "rerank": False},
    "hybrid": {"search_mode": "hybrid", "rerank": False},
    "hybrid-rerank20": {"search_mode": "hybrid", "rerank": True, "rerank_candidates": 20},
}
DEFAULT_SETUP = "hybrid-rerank20"  # what kb ships with
K = 5


def _rank(results: list[dict], item: dict) -> int | None:
    """Rank of the first result from the gold file whose lines overlap the gold chunk's lines."""
    for i, r in enumerate(results, 1):
        if r["source"] == item["source"] and r["start_line"] <= item["end_line"] and item["start_line"] <= r["end_line"]:
            return i
    return None


def evaluate(settings: Settings, gold: list[dict]) -> dict:
    answerable = [g for g in gold if g["answerable"]]
    unanswerable = [g for g in gold if not g["answerable"]]
    if not answerable:
        raise ValueError("gold set has no answerable questions")

    setups = {}
    for name, overrides in SETUPS.items():
        s = dataclasses.replace(settings, top_k=K, min_score=None, **overrides)
        search("warmup", s)
        rr, latency = [], []
        for g in answerable:
            t = time.perf_counter()
            rank = _rank(search(g["question"], s), g)
            latency.append((time.perf_counter() - t) * 1000)
            rr.append(1 / rank if rank else 0.0)
        setups[name] = {"hit@5": round(mean([x > 0 for x in rr]), 3), "mrr@5": round(mean(rr), 3),
                        "p50_ms": round(statistics.median(latency), 1), "per_question": rr}

    base = setups[DEFAULT_SETUP]["per_question"]
    for name, m in setups.items():
        m["p_vs_default"] = None if name == DEFAULT_SETUP else round(paired_randomization(base, m["per_question"]), 3)

    # abstention: how the default setup's top score separates answerable from unanswerable here
    s = dataclasses.replace(settings, top_k=1, min_score=None, **SETUPS[DEFAULT_SETUP])
    top = lambda q: (search(q, s) or [{"score": float("-inf")}])[0]["score"]  # noqa: E731
    pos, neg = [top(g["question"]) for g in answerable], [top(g["question"]) for g in unanswerable]
    abstention = {"n_answerable": len(pos), "n_unanswerable": len(neg)}
    shipped = Settings().min_score
    abstention["shipped_threshold"] = {
        "threshold": shipped,
        "answered_rate": round(mean([p >= shipped for p in pos]), 3),
        "abstained_rate": round(mean([n < shipped for n in neg]), 3) if neg else None,
    }
    if neg:
        for rule, kw in (("balanced", {}), ("min_answered_0.97", {"min_answered": 0.97})):
            abstention[rule] = {k: round(v, 3) for k, v in calibrate(pos, neg, **kw).items()}

    return {
        "date": datetime.date.today().isoformat(),
        "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "setups": setups, "abstention": abstention, "recommendation": _recommend(setups, len(answerable)),
    }


def _recommend(setups: dict, n: int) -> dict:
    best = max(setups, key=lambda k: (setups[k]["mrr@5"], k == DEFAULT_SETUP))
    p = setups[best]["p_vs_default"]
    if best != DEFAULT_SETUP and p is not None and p < 0.05:
        return {"setup": best, "reason": f"better than the default on your docs (MRR@5 {setups[best]['mrr@5']} vs "
                                         f"{setups[DEFAULT_SETUP]['mrr@5']}, p = {p})"}
    reason = "no setup is significantly better than the default on your docs"
    if n < 100:
        reason += f" (only {n} answerable questions; use --n 100 or more for a sharper comparison)"
    return {"setup": DEFAULT_SETUP, "reason": reason}


def format_report(r: dict) -> str:
    lines = [f"kb eval: {r['n_answerable']} answerable and {r['n_unanswerable']} unanswerable questions", "",
             f"{'setup':<17}{'hit@5':>7}{'MRR@5':>7}{'p50 ms':>8}{'p vs default':>14}"]
    for name, m in r["setups"].items():
        p = "default" if m["p_vs_default"] is None else f"{m['p_vs_default']:.3f}"
        lines.append(f"{name:<17}{m['hit@5']:>7.3f}{m['mrr@5']:>7.3f}{m['p50_ms']:>8}{p:>14}")
    a = r["abstention"]
    lines += ["", "Not-found threshold (reranker top score):"]
    sh = a["shipped_threshold"]
    refused = "n/a" if sh["abstained_rate"] is None else f"{sh['abstained_rate']:.0%}"
    lines.append(f"  {'shipped default':<17} {sh['threshold']:>7.2f}: answers {sh['answered_rate']:.0%}, refuses {refused}")
    for rule in ("balanced", "min_answered_0.97"):
        if rule in a:
            c = a[rule]
            lines.append(f"  {rule:<17} {c['threshold']:>7.2f}: answers {c['answered_rate']:.0%}, "
                         f"refuses {c['abstained_rate']:.0%}")
    rec = r["recommendation"]
    lines += ["", f"Recommendation: {rec['setup']}. {rec['reason'][0].upper() + rec['reason'][1:]}."]
    if "review_pass_rate" in r:
        lines.append(f"Human review: {r['review_pass_rate']:.0%} of sampled questions judged fair.")
    return "\n".join(lines)
