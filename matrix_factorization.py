"""Simplified rank-1 factorization. Squared error on ratings, not a status classifier.

This is the equation family of the Community Notes scorer with one factor.
It is not the production scorer: tag weights, helpfulness filters, and the
rest of that pipeline are absent. Rater terms are fit on train-split notes
only. A ranked note's intercept is then fit from its own pre-cutoff ratings
with those rater terms frozen. The fit-check RMSE uses train-split edges that
were removed before the fit. It is not a holdout ranking metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cn.config import (
    DATA_DIR,
    HOLDOUT_START,
    LAMBDA_F,
    LAMBDA_I,
    MF_EDGE_BATCH,
    MF_HOLDOUT_FRACTION,
    MF_MAX_EPOCHS,
    MF_TOL,
    MILESTONE_DIR,
    SEED,
)
from cn.early_features import aggregate_counts, coded_helpfulness, score_frame
from cn.metrics import (
    SCORER_STATUS,
    code_provenance,
    evaluate_by_stratum,
    evaluate_scores,
    note_tie_break,
    outcome_targets,
    rating_count_bin,
    write_json,
)


def hold_out_edges(ratings: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split edges before any parameter is fit. The held-out rows are not training rows."""
    rng = np.random.default_rng(seed)
    chosen = rng.random(len(ratings)) < fraction
    if len(ratings) and bool(chosen.all()):
        chosen = chosen.copy()
        chosen[0] = False
    return ratings.loc[~chosen].reset_index(drop=True), ratings.loc[chosen].reset_index(drop=True)


def edge_pairs(ratings: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(ratings["rater_id"].astype(str), ratings["note_id"].astype(str)))


def _ridge(design: np.ndarray, target: np.ndarray, penalties: tuple[float, float]) -> np.ndarray:
    gram = design.T @ design
    gram[0, 0] += penalties[0]
    gram[1, 1] += penalties[1]
    try:
        return np.linalg.solve(gram, design.T @ target)
    except np.linalg.LinAlgError:
        return np.zeros(2, dtype=np.float64)


def _grouped_update(
    group_ids: np.ndarray,
    other_factor: np.ndarray,
    target: np.ndarray,
    n_groups: int,
    penalties: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge intercept and factor for each group. Missing groups stay at 0.

    The 2x2 system is the same one `_ridge` solves, written with group sums so
    a million groups do not become a million Python calls.
    """
    intercept = np.zeros(n_groups, dtype=np.float64)
    factor = np.zeros(n_groups, dtype=np.float64)
    if len(group_ids) == 0:
        return intercept, factor
    counts = np.bincount(group_ids, minlength=n_groups).astype(np.float64)
    sum_other = np.bincount(group_ids, weights=other_factor, minlength=n_groups)
    sum_other_sq = np.bincount(group_ids, weights=other_factor * other_factor, minlength=n_groups)
    sum_target = np.bincount(group_ids, weights=target, minlength=n_groups)
    sum_cross = np.bincount(group_ids, weights=target * other_factor, minlength=n_groups)
    gram_00 = counts + penalties[0]
    gram_01 = sum_other
    gram_11 = sum_other_sq + penalties[1]
    det = gram_00 * gram_11 - gram_01 * gram_01
    present = (counts > 0) & (np.abs(det) >= 1e-12)
    safe = np.where(present, det, 1.0)
    intercept = np.where(present, (gram_11 * sum_target - gram_01 * sum_cross) / safe, 0.0)
    factor = np.where(present, (gram_00 * sum_cross - gram_01 * sum_target) / safe, 0.0)
    return intercept, factor


def fit_rater_terms(train: pd.DataFrame, seed: int = 0) -> dict:
    """Fit mu, rater terms, and train-split note terms by alternating least squares.

    One factor side starts from a seeded normal draw. Both sides at zero is a
    fixed point: each factor update multiplies the other side, so the product
    never leaves zero and the fit collapses to intercepts.
    """
    values = coded_helpfulness(train["helpfulness"]).to_numpy(dtype=np.float64)
    rater_ids = sorted(train["rater_id"].astype(str).unique())
    note_ids = sorted(train["note_id"].astype(str).unique())
    rater_index = {rater_id: i for i, rater_id in enumerate(rater_ids)}
    note_index = {note_id: i for i, note_id in enumerate(note_ids)}
    u = train["rater_id"].astype(str).map(rater_index).to_numpy()
    n = train["note_id"].astype(str).map(note_index).to_numpy()
    mu = float(values.mean()) if len(values) else 0.0
    i_u = np.zeros(len(rater_ids))
    f_u = np.random.default_rng(seed).normal(0.0, 0.1, size=len(rater_ids))
    i_n = np.zeros(len(note_ids))
    f_n = np.zeros(len(note_ids))
    previous = None
    epochs = 0
    penalties = (LAMBDA_I, LAMBDA_F)
    for epochs in range(1, MF_MAX_EPOCHS + 1):
        target_n = values - mu - i_u[u]
        i_n, f_n = _grouped_update(n, f_u[u], target_n, len(note_ids), penalties)
        target_u = values - mu - i_n[n]
        i_u, f_u = _grouped_update(u, f_n[n], target_u, len(rater_ids), penalties)
        mu = float(np.mean(values - i_u[u] - i_n[n] - f_u[u] * f_n[n]))
        prediction = mu + i_u[u] + i_n[n] + f_u[u] * f_n[n]
        rmse = float(np.sqrt(np.mean((values - prediction) ** 2)))
        if previous is not None and previous - rmse < MF_TOL:
            break
        previous = rmse
    return {
        "mu": mu,
        "epochs": epochs,
        "i_u": {rater_id: float(i_u[index]) for rater_id, index in rater_index.items()},
        "f_u": {rater_id: float(f_u[index]) for rater_id, index in rater_index.items()},
        "i_n": {note_id: float(i_n[index]) for note_id, index in note_index.items()},
        "f_n": {note_id: float(f_n[index]) for note_id, index in note_index.items()},
    }


def note_intercepts(ratings: pd.DataFrame, mu: float, i_u: dict, f_u: dict, note_ids: pd.Series) -> pd.Series:
    """Fit i_n with rater terms frozen. A note with no ratings stays at 0."""
    intercept = pd.Series(0.0, index=note_ids.astype(str).to_numpy())
    if len(ratings) == 0:
        return intercept
    frame = ratings.copy()
    frame["note_id"] = frame["note_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame = frame.loc[frame["note_id"].isin(intercept.index)]
    frame["value"] = coded_helpfulness(frame["helpfulness"])
    frame["i_u"] = frame["rater_id"].map(i_u).fillna(0.0)
    frame["f_u"] = frame["rater_id"].map(f_u).fillna(0.0)
    if len(frame) == 0:
        return intercept
    codes, uniques = pd.factorize(frame["note_id"], sort=False)
    target = (frame["value"] - mu - frame["i_u"]).to_numpy(dtype=np.float64)
    factor = frame["f_u"].to_numpy(dtype=np.float64)
    fitted, _unused = _grouped_update(codes.astype(np.int64), factor, target, len(uniques), (LAMBDA_I, LAMBDA_F))
    intercept.loc[uniques] = fitted
    return intercept


def predict(ratings: pd.DataFrame, fit: dict) -> np.ndarray:
    i_u = ratings["rater_id"].astype(str).map(fit["i_u"]).fillna(0.0).to_numpy(dtype=np.float64)
    f_u = ratings["rater_id"].astype(str).map(fit["f_u"]).fillna(0.0).to_numpy(dtype=np.float64)
    i_n = ratings["note_id"].astype(str).map(fit["i_n"]).fillna(0.0).to_numpy(dtype=np.float64)
    f_n = ratings["note_id"].astype(str).map(fit["f_n"]).fillna(0.0).to_numpy(dtype=np.float64)
    values = coded_helpfulness(ratings["helpfulness"]).to_numpy(dtype=np.float64)
    prediction = fit["mu"] + i_u + i_n + f_u * f_n
    return values, prediction


def rmse(ratings: pd.DataFrame, fit: dict) -> float:
    """Squared error in chunks of MF_EDGE_BATCH so the residual is not one array."""
    if len(ratings) == 0:
        return float("nan")
    total = 0.0
    count = 0
    for start in range(0, len(ratings), MF_EDGE_BATCH):
        values, prediction = predict(ratings.iloc[start : start + MF_EDGE_BATCH], fit)
        total += float(np.sum((values - prediction) ** 2))
        count += int(len(values))
    return float(np.sqrt(total / count))


def write_rank1_metrics(ratings: pd.DataFrame, window: pd.DataFrame, pool: pd.DataFrame, path: Path) -> dict:
    """Holdout ranking by i_n, plus a train-split held-out RMSE that is not that ranking."""
    holdout_ms = int(pd.Timestamp(HOLDOUT_START, tz="UTC").value // 10**6)
    created = window.copy()
    created["note_id"] = created["note_id"].astype(str)
    created_ms = created.set_index("note_id")["created_ms"]
    frame = ratings.copy()
    frame["note_id"] = frame["note_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame["created_ms"] = frame["note_id"].map(created_ms)
    fitting = frame.loc[frame["created_ms"] < holdout_ms]
    train, held_out = hold_out_edges(fitting, MF_HOLDOUT_FRACTION, seed=0)
    if edge_pairs(train) & edge_pairs(held_out):
        raise SystemExit("held-out edges were not removed before the fit")
    fit = fit_rater_terms(train, seed=SEED)
    holdout = pool.loc[pool["split"] == "holdout"].reset_index(drop=True)
    holdout_ratings = frame.loc[frame["note_id"].isin(set(holdout["note_id"].astype(str)))]
    intercepts = note_intercepts(holdout_ratings, fit["mu"], fit["i_u"], fit["f_u"], holdout["note_id"])
    scores = holdout["note_id"].astype(str).map(intercepts).fillna(0.0).to_numpy(dtype=np.float64)
    y_positive, y_labeled, hours = outcome_targets(holdout)
    strata = rating_count_bin(score_frame(holdout, aggregate_counts(frame))["early_count"].to_numpy())
    tie_break = note_tie_break(holdout["note_id"])
    factor_abs = max((abs(value) for value in fit["f_u"].values()), default=0.0)
    if factor_abs == 0.0:
        raise SystemExit("rank-1 factors stayed at zero")
    report = {
        "interpretation": (
            "Rank 1 is a simplified fit of the same equation family as the scorer, not the production scorer. "
            "An unrated note sits at intercept 0. The mean baseline scores that note as -1. "
            + SCORER_STATUS
        ),
        "holdout": evaluate_scores(y_positive, y_labeled, scores, hours, tie_break=tie_break),
        "holdout_by_rating_count": evaluate_by_stratum(
            y_positive, y_labeled, scores, hours, strata, tie_break=tie_break
        ),
        "fit_check": {
            "rmse_held_out": rmse(held_out, fit),
            "n_held_out": int(len(held_out)),
            "n_train": int(len(train)),
            "epochs": fit["epochs"],
            "max_abs_rater_factor": factor_abs,
            "note": "Training-edge RMSE is not reported. This number is not a holdout ranking metric.",
        },
        "provenance": code_provenance(),
    }
    write_json(path, report)
    return report


def main() -> None:
    ratings = pd.read_parquet(DATA_DIR / "ratings_pre_cutoff_12h.parquet")
    window = pd.read_parquet(DATA_DIR / "window_notes.parquet")
    pool = pd.read_parquet(DATA_DIR / "pool_12h.parquet")
    path = MILESTONE_DIR / "rank1_metrics.json"
    report = write_rank1_metrics(ratings, window, pool, path)
    print(f"wrote {path}")
    print("holdout recall@1%", report["holdout"]["at_frac"]["0.01"]["recall"])
    print("held-out rmse", report["fit_check"]["rmse_held_out"])


if __name__ == "__main__":
    main()
