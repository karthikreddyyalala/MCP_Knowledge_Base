from mcp_kb.evals.abstain import calibrate


def test_separable_scores_give_perfect_balanced_accuracy():
    r = calibrate(pos=[-1.0, 0.5, 2.0], neg=[-11.0, -9.0, -6.0])
    assert -6.0 < r["threshold"] <= -1.0
    assert r["balanced_accuracy"] == 1.0
    assert r["answered_rate"] == 1.0 and r["abstained_rate"] == 1.0


def test_overlapping_scores_pick_the_best_cut():
    r = calibrate(pos=[-3.0, 1.0, 2.0, 3.0], neg=[-10.0, -8.0, -2.0, -9.0])
    # cutting at 1.0 keeps 3/4 answerable and rejects 4/4 unanswerable -> 0.875 balanced accuracy
    assert r["balanced_accuracy"] == 0.875
