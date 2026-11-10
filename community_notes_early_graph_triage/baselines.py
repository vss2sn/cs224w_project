"""Score functions that do not use a trained graph model.

``newest_first_scores`` and ``random_scores`` need only the status table.
``rating_count_scores`` and ``mean_helpfulness_scores`` are the baselines that
wait on the ratings stream: pass the early count and the early mean.
"""

from __future__ import annotations

import numpy as np


def newest_first_scores(created_ms: np.ndarray) -> np.ndarray:
    """Higher means newer, which is the unresolved-tab order."""
    return np.asarray(created_ms, dtype=np.float64)


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def rating_count_scores(counts: np.ndarray) -> np.ndarray:
    """Rank notes that already have more ratings above notes that have fewer."""
    return np.asarray(counts, dtype=np.float64)


def mean_helpfulness_scores(means: np.ndarray) -> np.ndarray:
    """Rank by the mean of ratings observed at or before the cutoff.

    Encode Helpful as 1, Somewhat Helpful as 0.5, and Not Helpful as 0, the
    same coding as the production factorization.
    """
    return np.asarray(means, dtype=np.float64)
