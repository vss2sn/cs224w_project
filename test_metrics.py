"""Synthetic checks for the ranking metrics. These do not read the snapshot."""

import numpy as np

from cn.baselines import mean_helpfulness_scores, newest_first_scores, rating_count_scores
from cn.metrics import (
    average_precision,
    evaluate_by_stratum,
    evaluate_scores,
    ndcg_at_k,
    rating_count_bin,
    recall_at_k,
)


def test_perfect_ranking_captures_the_budget():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([9, 8, 7, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    # 10% of 10 notes is 1 slot, and the top slot is a positive: recall 1/3.
    assert abs(recall_at_k(y, scores, 1) - 1 / 3) < 1e-12
    assert ndcg_at_k(y, scores, 1) == 1.0
    assert average_precision(y, scores) == 1.0


def test_reversed_ranking_captures_nothing_in_the_first_slot():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0, 3, 2, 1], dtype=float)
    assert recall_at_k(y, scores, 1) == 0.0
    assert ndcg_at_k(y, scores, 1) == 0.0


def test_evaluate_scores_uses_a_fraction_of_the_pool():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    labeled = np.array([1, 1, 0, 0, -1, -1, -1, -1, -1, -1])
    scores = np.array([5, 4, 3, 2, 0, 0, 0, 0, 0, 0], dtype=float)
    hours = np.array([20, 30, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    report = evaluate_scores(y, labeled, scores, hours, fractions=(0.2,))
    assert report["at_frac"]["0.20"]["k"] == 2
    assert report["at_frac"]["0.20"]["positives_captured"] == 2
    assert report["at_frac"]["0.20"]["recall"] == 1.0
    assert report["at_frac"]["0.20"]["median_hours_among_captured"] == 25.0
    assert report["auprc_labeled"] == 1.0


def test_stratum_k_is_a_fraction_of_the_bin():
    y = np.array([1, 0, 1, 0])
    labeled = np.array([1, 0, 1, 0])
    scores = np.array([2, 1, 0, 3], dtype=float)
    hours = np.array([10, np.nan, 12, np.nan])
    strata = np.array(["0", "0", "10+", "10+"])
    report = evaluate_by_stratum(y, labeled, scores, hours, strata, fractions=(0.5,))
    assert report["0"]["n_pool"] == 2
    assert report["0"]["at_frac"]["0.50"]["k"] == 1
    assert set(rating_count_bin(np.array([0, 1, 3, 7, 12]))) == {"0", "1", "2-4", "5-9", "10+"}


def test_note_tie_break_is_stable_and_not_creation_order():
    import pandas as pd

    from cn.metrics import note_tie_break

    ids = pd.Series([str(10**18 + i) for i in range(20)])
    first = note_tie_break(ids)
    assert np.array_equal(first, note_tie_break(ids))
    # These ids increase with creation time. The hash must not follow that order.
    assert not np.all(np.diff(first) > 0)


def test_tied_boundary_reports_the_range_not_the_file_order():
    from cn.metrics import tie_capture

    y = np.array([1, 0, 0])
    scores = np.array([1.0, 1.0, 1.0])
    captured = tie_capture(scores, y, k=1)
    assert captured["positives_min"] == 0
    assert captured["positives_max"] == 1
    assert abs(captured["positives_expected"] - (1 / 3)) < 1e-12
    # Reversing the rows must not change the expected recall.
    assert abs(recall_at_k(y[::-1], scores, 1) - recall_at_k(y, scores, 1)) < 1e-12


def test_baseline_score_directions():
    created = np.array([10, 30, 20])
    assert list(np.argsort(-newest_first_scores(created))) == [1, 2, 0]
    assert list(rating_count_scores(np.array([1, 4]))) == [1.0, 4.0]
    assert list(mean_helpfulness_scores(np.array([0.0, 0.5, 1.0]))) == [0.0, 0.5, 1.0]
