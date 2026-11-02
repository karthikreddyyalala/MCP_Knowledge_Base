"""Significance testing for paired per-query scores."""
import random


def paired_randomization(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value that b's mean differs from a's by chance.

    Under the null hypothesis the two systems are interchangeable, so each query's
    difference is equally likely to have either sign. We flip signs at random n times
    and count how often the mean difference is at least as large as the observed one.
    """
    diffs = [y - x for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(n))
    return (extreme + 1) / (n + 1)  # +1 so p is never exactly 0 from a finite sample
