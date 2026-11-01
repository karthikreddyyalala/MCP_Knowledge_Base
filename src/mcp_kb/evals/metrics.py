"""Retrieval metrics, matching BEIR / pytrec_eval definitions.

ranked: doc ids in rank order (best first).
qrels:  doc id -> relevance grade for one query. Grade 0 means not relevant.
"""
import math


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Rewards putting the most relevant docs highest. 1.0 = ideal order."""
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Share of the relevant docs that appear anywhere in the top k."""
    relevant = {d for d, g in qrels.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant doc; 0 if none in the top k."""
    for i, d in enumerate(ranked[:k]):
        if qrels.get(d, 0) > 0:
            return 1 / (i + 1)
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
