"""Pick the "not found" threshold on the reranker's top score."""


def calibrate(pos: list[float], neg: list[float]) -> dict:
    """pos: top scores for answerable queries; neg: top scores for unanswerable ones.

    Tries every observed score as the cut (answer if top score >= cut) and keeps the one with
    the best balanced accuracy: the mean of "answerable queries answered" and "unanswerable
    queries refused", so neither side wins just by being the larger set.
    """
    best = None
    for t in sorted(set(pos) | set(neg)):
        answered = sum(p >= t for p in pos) / len(pos)
        abstained = sum(n < t for n in neg) / len(neg)
        ba = (answered + abstained) / 2
        if best is None or ba > best["balanced_accuracy"]:
            best = {"threshold": t, "balanced_accuracy": ba, "answered_rate": answered, "abstained_rate": abstained}
    return best
