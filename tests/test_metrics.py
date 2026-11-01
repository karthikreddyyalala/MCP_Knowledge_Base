import math

from mcp_kb.evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_perfect_ranking_scores_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == 1.0
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == 1.0
    assert mrr_at_k(["a", "b"], {"a": 1}, 10) == 1.0


def test_ndcg_hand_computed():
    # relevant doc at rank 2 only: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert math.isclose(ndcg_at_k(["x", "a"], {"a": 1}, 10), 1 / math.log2(3))


def test_ndcg_uses_graded_gain():
    # grades 2 and 1, returned in the wrong order
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10), dcg / idcg)


def test_cutoff_and_zero_grades_are_ignored():
    assert ndcg_at_k(["x"] * 10 + ["a"], {"a": 1}, 10) == 0.0
    assert recall_at_k(["a"], {"a": 1, "b": 0}, 10) == 1.0  # grade 0 is not relevant
    assert mrr_at_k(["x", "x", "a"], {"a": 1}, 10) == 1 / 3


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a": 1, "b": 1}, 10) == 0.5


def test_spearman():
    from mcp_kb.evals.validate import spearman

    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    # human order dense<bm25<hybrid<rerank vs synthetic bm25 first: rho = 1 - 6*6/(4*15) = 0.4
    assert abs(spearman([0.659, 0.685, 0.711, 0.720], [0.579, 0.869, 0.741, 0.804]) - 0.4) < 1e-9
