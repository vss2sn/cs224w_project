#!/usr/bin/env python3
"""Score newest-first and random on the study window.

Rating-count and mean-helpfulness need the ratings stream, so this script
does not run them. It writes the metric report the milestone cites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.baselines import newest_first_scores, random_scores
from community_notes_early_graph_triage.config import CUTOFF_H, DATA_DIR, HOLDOUT_START, MILESTONE_DIR, SNAPSHOT, WINDOW_END, WINDOW_START
from community_notes_early_graph_triage.labels import DROPPED, NEGATIVE, POSITIVE, RETURNED, UNRESOLVED
from community_notes_early_graph_triage.metrics import POOL_FRACTIONS, code_provenance, evaluate_scores, note_tie_break, outcome_targets

OUT_PATH = MILESTONE_DIR / "baseline_metrics.json"
RANDOM_SEEDS = range(20)


def targets(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return outcome_targets(pool)


def mean_reports(reports: list[dict]) -> dict:
    """Average the random-seed reports. K and the counts of notes stay fixed."""
    fractions = list(reports[0]["at_frac"])
    at_frac = {}
    for fraction in fractions:
        keys = ("recall", "ndcg", "positives_captured", "median_hours_among_captured")
        at_frac[fraction] = {"k": reports[0]["at_frac"][fraction]["k"]}
        for key in keys:
            values = [report["at_frac"][fraction][key] for report in reports]
            at_frac[fraction][key] = float(np.nanmean(values))
    return {
        "n_pool": reports[0]["n_pool"],
        "n_positive": reports[0]["n_positive"],
        "n_labeled_pair": reports[0]["n_labeled_pair"],
        "auprc_labeled": float(np.nanmean([report["auprc_labeled"] for report in reports])),
        "auprc_pool": float(np.nanmean([report["auprc_pool"] for report in reports])),
        "at_frac": at_frac,
        "n_seeds": len(reports),
    }


def gain(model: dict, baseline: dict) -> dict:
    """Model minus baseline at each budget. Positive recall means more Helpful notes."""
    out = {}
    for fraction, row in model["at_frac"].items():
        base = baseline["at_frac"][fraction]
        out[fraction] = {
            "recall": row["recall"] - base["recall"],
            "ndcg": row["ndcg"] - base["ndcg"],
            "positives_captured": row["positives_captured"] - base["positives_captured"],
            "median_hours_among_captured": row["median_hours_among_captured"] - base["median_hours_among_captured"],
        }
    out["auprc_labeled"] = model["auprc_labeled"] - baseline["auprc_labeled"]
    out["auprc_pool"] = model["auprc_pool"] - baseline["auprc_pool"]
    return out


def evaluate_split(pool: pd.DataFrame) -> dict:
    y_positive, y_labeled, hours = targets(pool)
    created = pool["created_ms"].to_numpy()
    tie_break = note_tie_break(pool["note_id"])
    newest = evaluate_scores(y_positive, y_labeled, newest_first_scores(created), hours, tie_break=tie_break)
    random_runs = [
        evaluate_scores(y_positive, y_labeled, random_scores(len(pool), seed), hours, tie_break=tie_break)
        for seed in RANDOM_SEEDS
    ]
    random_mean = mean_reports(random_runs)
    counts = pool["outcome"].value_counts().to_dict()
    return {
        "n_pool": int(len(pool)),
        "outcomes": {name: int(counts.get(name, 0)) for name in (POSITIVE, NEGATIVE, DROPPED, RETURNED, UNRESOLVED)},
        "prevalence_in_pool": float(y_positive.mean()),
        "prevalence_in_labeled_pair": float(y_positive[y_labeled >= 0].mean()) if (y_labeled >= 0).any() else None,
        "newest_first": newest,
        "random_mean_20_seeds": random_mean,
        "newest_minus_random": gain(newest, random_mean),
    }


def main() -> None:
    pool_path = DATA_DIR / "pool_12h.parquet"
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}; run note_tables first")
    pool = pd.read_parquet(pool_path)
    pool["note_id"] = pool["note_id"].astype(str)
    holdout = pool.loc[pool["split"] == "holdout"].reset_index(drop=True)
    report = {
        "snapshot": SNAPSHOT,
        "window": f"{WINDOW_START} to {WINDOW_END}",
        "holdout_window": f"{HOLDOUT_START} to {WINDOW_END}",
        "cutoff_h": CUTOFF_H,
        "fractions_of_pool": list(POOL_FRACTIONS),
        "full_window": evaluate_split(pool),
        "holdout": evaluate_split(holdout),
        "provenance": code_provenance(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    full = report["full_window"]
    print(f"wrote {OUT_PATH}")
    print("pool", full["n_pool"], "outcomes", full["outcomes"])
    print(
        "prevalence pool",
        round(full["prevalence_in_pool"], 4),
        "labeled pair",
        round(full["prevalence_in_labeled_pair"], 4),
    )
    for name in ("newest_first", "random_mean_20_seeds"):
        row = full[name]
        top = row["at_frac"]["0.01"]
        print(
            f"{name}: auprc_labeled {row['auprc_labeled']:.4f}  auprc_pool {row['auprc_pool']:.4f}  "
            f"recall@1% {top['recall']:.4f}  ndcg@1% {top['ndcg']:.4f}  "
            f"positives {top['positives_captured']:.1f}  median_hours {top['median_hours_among_captured']:.2f}"
        )


if __name__ == "__main__":
    main()
