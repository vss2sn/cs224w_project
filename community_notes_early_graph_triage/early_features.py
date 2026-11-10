"""Early rating count and mean on the 12-hour queue.

A note with no pre-cutoff ratings scores -1 on the mean ranking so it sorts
after every real mean. That -1 is a ranking policy. The graph stores 0 for the
same note, which is a feature the head can read, not a score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.baselines import mean_helpfulness_scores, rating_count_scores
from community_notes_early_graph_triage.config import DATA_DIR, HELPFULNESS_CODE, MILESTONE_DIR
from community_notes_early_graph_triage.metrics import (
    SCORER_STATUS,
    code_provenance,
    evaluate_by_stratum,
    evaluate_scores,
    note_tie_break,
    outcome_targets,
    rating_count_bin,
    write_json,
)


def coded_helpfulness(helpfulness: pd.Series) -> pd.Series:
    """Map HELPFUL / SOMEWHAT_HELPFUL / NOT_HELPFUL to 1 / 0.5 / 0."""
    return helpfulness.map(HELPFULNESS_CODE).astype(float)


def aggregate_counts(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-note count and mean. The mean is the average of the coded ratings."""
    values = coded_helpfulness(ratings["helpfulness"])
    grouped = ratings.assign(value=values).groupby("note_id", sort=False)["value"]
    return grouped.agg(early_count="size", early_mean="mean").reset_index()


def score_frame(pool: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Align count and mean to pool order. Missing notes get count 0 and mean score -1."""
    merged = pool[["note_id"]].merge(counts, on="note_id", how="left")
    count = merged["early_count"].fillna(0).to_numpy(dtype=np.float64)
    mean = merged["early_mean"].to_numpy(dtype=np.float64)
    # A missing mean is scored -1 so it sorts after every observed mean in [0, 1].
    mean_score = np.where(count > 0, mean, -1.0)
    return pd.DataFrame({"early_count": count, "early_mean": mean, "mean_score": mean_score})


def _report(pool: pd.DataFrame, scores: np.ndarray, strata: np.ndarray) -> dict:
    y_positive, y_labeled, hours = outcome_targets(pool)
    tie_break = note_tie_break(pool["note_id"])
    return {
        "scores": evaluate_scores(y_positive, y_labeled, scores, hours, tie_break=tie_break),
        "by_rating_count": evaluate_by_stratum(
            y_positive, y_labeled, scores, hours, strata, tie_break=tie_break
        ),
    }


def metrics_for(pool: pd.DataFrame, counts: pd.DataFrame) -> dict:
    """Count and mean metrics for one pool, including rating-count bins."""
    frame = score_frame(pool, counts)
    strata = rating_count_bin(frame["early_count"].to_numpy())
    return {
        "rating_count": _report(pool, rating_count_scores(frame["early_count"]), strata),
        "mean_helpfulness": _report(pool, mean_helpfulness_scores(frame["mean_score"]), strata),
    }


def write_early_feature_metrics(ratings: pd.DataFrame, pool: pd.DataFrame, path: Path) -> dict:
    counts = aggregate_counts(ratings)
    report = {
        "interpretation": SCORER_STATUS,
        "unrated_note": (
            "The mean baseline scores a note with no ratings as -1 and sorts it last. "
            "Rank 1 leaves that note at intercept 0. Those are different policies, and "
            "rating-count bin 0 is where they disagree."
        ),
        "full_pool": metrics_for(pool, counts),
        "holdout": metrics_for(pool.loc[pool["split"] == "holdout"].reset_index(drop=True), counts),
        "provenance": code_provenance(),
    }
    write_json(path, report)
    return report


def main() -> None:
    ratings_path = DATA_DIR / "ratings_pre_cutoff_12h.parquet"
    pool_path = DATA_DIR / "pool_12h.parquet"
    if not ratings_path.exists() or not pool_path.exists():
        raise SystemExit("missing ratings or pool parquet")
    ratings = pd.read_parquet(ratings_path, columns=["note_id", "helpfulness"])
    ratings["note_id"] = ratings["note_id"].astype(str)
    pool = pd.read_parquet(pool_path)
    pool["note_id"] = pool["note_id"].astype(str)
    path = MILESTONE_DIR / "early_feature_metrics.json"
    report = write_early_feature_metrics(ratings, pool, path)
    holdout = report["holdout"]["rating_count"]["scores"]
    print(f"wrote {path}")
    print("holdout count recall@1%", holdout["at_frac"]["0.01"]["recall"])


if __name__ == "__main__":
    main()
