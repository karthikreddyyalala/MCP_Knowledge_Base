"""Reciprocal Rank Fusion: merge ranked lists using rank positions, not raw scores."""


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Each list gives an item 1 / (k + rank). Items ranked well by several lists rise to the top.

    Ranks are used because dense (cosine) and BM25 scores are on unrelated scales;
    k = 60 is the constant from the original paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
