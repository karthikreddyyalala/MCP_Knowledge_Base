from mcp_kb.evals.stats import paired_randomization
from mcp_kb.fusion import rrf


def test_identical_runs_are_not_significant():
    a = [0.1, 0.5, 0.9, 0.3]
    assert paired_randomization(a, a) == 1.0


def test_consistent_improvement_is_significant():
    a = [i / 100 for i in range(50)]
    b = [x + 0.2 for x in a]
    assert paired_randomization(a, b) < 0.01


def test_rrf_rewards_agreement():
    # a = 1/61 + 1/62 = 0.03252, c = 1/63 + 1/61 = 0.03226, b = 1/62
    assert rrf([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]
