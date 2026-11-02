"""Pick the "not found" threshold on the reranker's top score."""


def calibrate(pos: list[float], neg: list[float], min_answered: float | None = None) -> dict:
    """pos: top scores for answerable queries; neg: top scores for unanswerable ones.

    Tries every observed score as the cut (answer if top score >= cut) and keeps the one with
    the best balanced accuracy: the mean of "answerable queries answered" and "unanswerable
    queries refused", so neither side wins just by being the larger set.

    With min_answered (e.g. 0.97), it instead refuses as many unanswerable queries as possible
    while still answering at least that share of answerable ones: "refuse only when confident".
    """
    best = None
    for t in sorted(set(pos) | set(neg)):
        answered = sum(p >= t for p in pos) / len(pos)
        abstained = sum(n < t for n in neg) / len(neg)
        ba = (answered + abstained) / 2
        if min_answered is not None:
            if answered < min_answered:
                continue
            better = best is None or abstained > best["abstained_rate"]
        else:
            better = best is None or ba > best["balanced_accuracy"]
        if better:
            best = {"threshold": t, "balanced_accuracy": ba, "answered_rate": answered, "abstained_rate": abstained}
    return best
