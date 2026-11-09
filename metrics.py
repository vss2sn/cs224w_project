"""Ranking metrics for the rater work queue.

The pool is every note still Needs More Ratings at the cutoff. A positive is
a note that later ends Currently Rated Helpful. K is a fraction of that pool,
matching the proposal's budgets of 1–10%.

Two average-precision numbers are reported because they answer different
questions:

- ``auprc_labeled`` ranks only positives against negatives (never shown, and
  the current status is Not Helpful). Dropped, returned, and still-unresolved
  notes are left out of this list.
- ``auprc_pool`` ranks the whole pool and treats every non-positive as 0. The
  base rate is the share of the pool that ends Helpful.

Recall@K and NDCG@K always use the whole pool. Recall's denominator is the
number of positives, so a slot spent on an unresolved note lowers recall.

Ties are not broken by file order. Recall@K is the expected count when the
slots on a tied boundary are drawn uniformly. NDCG@K with a tie of more than
one note is the mean of 32 shuffles from seed 0. Median hours uses a seeded
hash of the note id. Sorting the id would prefer older notes, because a
Community Notes id encodes creation time.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from cn.labels import NEGATIVE, POSITIVE

POOL_FRACTIONS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)

SCORER_STATUS = (
    "Recall@K is the share of notes whose snapshot status is CRH. "
    "It is not a measure of usefulness to readers, and it is not a count of notes that stayed Helpful."
)


def outcome_targets(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive, labeled-pair, and hours arrays for evaluate_scores.

    ``y_labeled`` is 1 for positives, 0 for negatives, and -1 for every other outcome.
    """
    outcome = pool["outcome"].to_numpy()
    y_positive = (outcome == POSITIVE).astype(np.int8)
    y_labeled = np.full(len(pool), -1, dtype=np.int8)
    y_labeled[outcome == POSITIVE] = 1
    y_labeled[outcome == NEGATIVE] = 0
    hours = pool["hours_to_resolution"].to_numpy(dtype=np.float64)
    return y_positive, y_labeled, hours


def align_scores(graph_note_ids: np.ndarray, scores: np.ndarray, pool_note_ids: np.ndarray) -> np.ndarray:
    """Place graph-order scores onto pool rows by note_id.

    Graph order includes notes that had already left Needs More Ratings, so it
    is not pool order. A degree-0 note still has a score and is kept.
    """
    index = {note_id: i for i, note_id in enumerate(graph_note_ids)}
    out = np.empty(len(pool_note_ids), dtype=np.float64)
    for i, note_id in enumerate(pool_note_ids):
        try:
            out[i] = scores[index[str(note_id)]]
        except KeyError as exc:
            raise KeyError(f"pool note {note_id} is missing from the graph") from exc
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def rating_count_bin(counts: np.ndarray) -> np.ndarray:
    """Bins for the early-rating-count stratification. Unused until ratings are joined."""
    counts = np.asarray(counts)
    bins = np.full(counts.shape, "10+", dtype=object)
    bins[counts <= 0] = "0"
    bins[counts == 1] = "1"
    bins[(counts >= 2) & (counts <= 4)] = "2-4"
    bins[(counts >= 5) & (counts <= 9)] = "5-9"
    return bins


def note_tie_break(note_ids: pd.Series) -> np.ndarray:
    """Seeded hash of the note id, for median hours and for a unique top K.

    The id is a snowflake, so the smaller id is the older note. This hash does
    not follow that order. Recall does not use it: a tied boundary is the
    expected count. NDCG uses it only when the boundary tie has one note.
    """
    codes = np.empty(len(note_ids), dtype=np.int64)
    for i, note_id in enumerate(note_ids.astype(str)):
        digest = hashlib.sha256(b"cn-tie-0\0" + note_id.encode()).digest()
        codes[i] = int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF
    return codes


def _order(scores: np.ndarray, tie_break: np.ndarray | None = None) -> np.ndarray:
    """Higher score first. Ties break by `tie_break`, which defaults to row index."""
    scores = np.asarray(scores, dtype=np.float64)
    if tie_break is None:
        tie_break = np.arange(len(scores))
    return np.lexsort((np.asarray(tie_break), -scores))


def tie_capture(scores: np.ndarray, y_true: np.ndarray, k: int) -> dict:
    """How many positives the top K can hold when the boundary score is tied.

    The expected count draws the remaining slots uniformly from the tie.
    The min and max are the worst and best subsets of that same tie.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(np.int8)
    k = min(int(k), len(scores))
    if k <= 0:
        return {
            "boundary_score": None,
            "n_tied": 0,
            "slots_from_tie": 0,
            "positives_min": 0,
            "positives_max": 0,
            "positives_expected": 0.0,
        }
    boundary = float(np.partition(scores, len(scores) - k)[len(scores) - k])
    above = scores > boundary
    tied = scores == boundary
    n_above = int(above.sum())
    n_tie = int(tied.sum())
    slots = k - n_above
    pos_above = int(y_true[above].sum())
    pos_tie = int(y_true[tied].sum())
    expected = pos_above + (slots * pos_tie / n_tie if n_tie else 0.0)
    low = pos_above + max(0, slots - (n_tie - pos_tie))
    high = pos_above + min(slots, pos_tie)
    return {
        "boundary_score": boundary,
        "n_tied": n_tie,
        "slots_from_tie": int(slots),
        "positives_min": int(low),
        "positives_max": int(high),
        "positives_expected": float(expected),
    }


def _k_for(n_pool: int, fraction: float) -> int:
    return max(1, int(round(fraction * n_pool)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int8)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(scores, dtype=np.float64)))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Expected recall when a tied boundary is broken uniformly at random."""
    y_true = np.asarray(y_true).astype(np.int8)
    total = int(y_true.sum())
    if total == 0:
        return float("nan")
    return tie_capture(scores, y_true, k)["positives_expected"] / total


def _ndcg_from_order(y_true: np.ndarray, order: np.ndarray, k: int) -> float:
    gained = y_true[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gained * discounts))
    n_rel = int(y_true.sum())
    idcg = float(np.sum(discounts[: min(n_rel, k)]))
    return dcg / idcg


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int, tie_break: np.ndarray | None = None) -> float:
    """Binary NDCG@K. A relevant item contributes 1/log2(rank+1).

    A unique top K uses `tie_break`. A tied boundary is the mean of 32 seeded
    shuffles of that tie, so the number does not follow the file order.
    """
    y_true = np.asarray(y_true).astype(np.float64)
    k = min(int(k), len(y_true))
    n_rel = int(y_true.sum())
    if k <= 0 or n_rel == 0:
        return float("nan")
    captured = tie_capture(scores, y_true, k)
    if captured["n_tied"] <= 1:
        return _ndcg_from_order(y_true, _order(scores, tie_break), k)
    rng = np.random.default_rng(0)
    samples = [
        _ndcg_from_order(y_true, _order(scores, rng.random(len(scores))), k) for _ in range(32)
    ]
    return float(np.mean(samples))


def positives_and_hours_at_k(
    y_true: np.ndarray,
    hours: np.ndarray,
    scores: np.ndarray,
    k: int,
    tie_break: np.ndarray | None = None,
) -> tuple[int, float]:
    """Positives and median hours in the top K under the declared tie break."""
    y_true = np.asarray(y_true).astype(np.int8)
    hours = np.asarray(hours, dtype=np.float64)
    chosen = _order(scores, tie_break)[: min(int(k), len(y_true))]
    captured = chosen[y_true[chosen] == 1]
    if len(captured) == 0:
        return 0, float("nan")
    return int(len(captured)), float(np.median(hours[captured]))


def evaluate_scores(
    y_positive: np.ndarray,
    y_labeled: np.ndarray,
    scores: np.ndarray,
    hours: np.ndarray,
    fractions: tuple[float, ...] = POOL_FRACTIONS,
    tie_break: np.ndarray | None = None,
) -> dict:
    """Metrics for one score vector.

    ``y_positive`` is 1 for notes that end Helpful, else 0, on the full pool.
    ``y_labeled`` is 1 for positives, 0 for negatives, and -1 for notes left
    out of the labeled-pair average precision (dropped, returned, unresolved).
    Recall is the expected count under a uniform draw from a tied boundary.
    Median hours uses ``tie_break``. Callers pass ``note_tie_break``, a seeded
    hash of the note id, so equal scores do not sort older notes first.
    """
    y_positive = np.asarray(y_positive).astype(np.int8)
    y_labeled = np.asarray(y_labeled).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    hours = np.asarray(hours, dtype=np.float64)
    n_pool = int(len(y_positive))
    labeled = y_labeled >= 0
    total_positive = int(y_positive.sum())
    at_frac = {}
    for fraction in fractions:
        k = _k_for(n_pool, fraction)
        captured = tie_capture(scores, y_positive, k)
        _n_pos, median_hours = positives_and_hours_at_k(y_positive, hours, scores, k, tie_break)
        expected = captured["positives_expected"]
        at_frac[f"{fraction:.2f}"] = {
            "k": k,
            "recall": expected / total_positive if total_positive else None,
            "recall_min": captured["positives_min"] / total_positive if total_positive else None,
            "recall_max": captured["positives_max"] / total_positive if total_positive else None,
            "ndcg": ndcg_at_k(y_positive, scores, k, tie_break),
            "positives_captured": expected,
            "positives_captured_min": captured["positives_min"],
            "positives_captured_max": captured["positives_max"],
            "n_tied_at_boundary": captured["n_tied"],
            "median_hours_among_captured": median_hours,
        }
    return {
        "n_pool": n_pool,
        "n_positive": int(y_positive.sum()),
        "n_labeled_pair": int(labeled.sum()),
        "auprc_labeled": average_precision(y_labeled[labeled], scores[labeled]),
        "auprc_pool": average_precision(y_positive, scores),
        "at_frac": at_frac,
    }


def evaluate_by_stratum(
    y_positive: np.ndarray,
    y_labeled: np.ndarray,
    scores: np.ndarray,
    hours: np.ndarray,
    strata: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
    tie_break: np.ndarray | None = None,
) -> dict:
    """Same metrics inside each early-rating-count bin. K is a fraction of the bin."""
    strata = np.asarray(strata)
    if tie_break is not None:
        tie_break = np.asarray(tie_break)
    out = {}
    for name in sorted(set(strata.tolist())):
        mask = strata == name
        out[str(name)] = evaluate_scores(
            y_positive[mask],
            y_labeled[mask],
            scores[mask],
            hours[mask],
            fractions,
            None if tie_break is None else tie_break[mask],
        )
    return out


def code_provenance() -> dict:
    """Git revision and a hash of project/cn, so a metric file names the code that wrote it."""
    from cn.config import ADAM_LR, ENCODER_HOPS, ENCODER_STEPS, ENCODER_WIDTH, SEED

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    repo = root.parents[1]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {
        "git_head": head.stdout.strip() if head.returncode == 0 else None,
        "cn_sha256": digest.hexdigest(),
        "seed": SEED,
        "adam_lr": ADAM_LR,
        "encoder_width": ENCODER_WIDTH,
        "encoder_hops": ENCODER_HOPS,
        "encoder_steps": ENCODER_STEPS,
    }
