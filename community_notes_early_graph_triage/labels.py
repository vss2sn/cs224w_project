"""Labels for notes that are still Needs More Ratings at a cutoff.

A note is still unresolved at cutoff t when its first non-NMR status is
missing or falls after createdAt + t. Status history does not store every
later flip, so a note that left and returned before t is not in the pool.

Outcomes for notes in the pool partition the pool:

- positive: current status is CURRENTLY_RATED_HELPFUL
- negative: never reached Helpful, current status is CURRENTLY_RATED_NOT_HELPFUL
- dropped: reached Helpful, current status is no longer Helpful
- returned: never reached Helpful, current status is Needs More Ratings again
- unresolved: no first non-NMR status by the snapshot
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HOUR_MS = 60 * 60 * 1000

CRH = "CURRENTLY_RATED_HELPFUL"
CRNH = "CURRENTLY_RATED_NOT_HELPFUL"
NMR = "NEEDS_MORE_RATINGS"

POSITIVE = "positive"
NEGATIVE = "negative"
DROPPED = "dropped"
RETURNED = "returned"
UNRESOLVED = "unresolved"
OTHER = "other"

WANTED = [
    "noteId",
    "createdAtMillis",
    "timestampMillisOfFirstNonNMRStatus",
    "firstNonNMRStatus",
    "currentStatus",
    "timestampMillisOfCurrentStatus",
    "timestampMillisOfLatestNonNMRStatus",
    "latestNonNMRStatus",
    "mostRecentNonNMRStatus",
    "lockedStatus",
    "timestampMillisOfStatusLock",
]


def load_status(path: Path) -> pd.DataFrame:
    """Read the columns the labeler needs from a note-status-history zip or TSV."""
    header = pd.read_csv(path, sep="\t", nrows=0, compression="zip").columns.tolist()
    usecols = [c for c in WANTED if c in header]
    required = (
        "noteId",
        "createdAtMillis",
        "timestampMillisOfFirstNonNMRStatus",
        "firstNonNMRStatus",
        "currentStatus",
    )
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"status file is missing {missing}; header={header}")
    frame = pd.read_csv(
        path,
        sep="\t",
        compression="zip",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
    )
    for col in (
        "createdAtMillis",
        "timestampMillisOfFirstNonNMRStatus",
        "timestampMillisOfCurrentStatus",
        "timestampMillisOfLatestNonNMRStatus",
        "timestampMillisOfStatusLock",
    ):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            # Empty, 0, and -1 mean "no such event" in this release.
            frame[col] = values.mask(values <= 0)
    if "latestNonNMRStatus" not in frame.columns and "mostRecentNonNMRStatus" in frame.columns:
        frame["latestNonNMRStatus"] = frame["mostRecentNonNMRStatus"]
    frame["created"] = pd.to_datetime(frame["createdAtMillis"], unit="ms", utc=True)
    frame["month"] = frame["created"].dt.strftime("%Y-%m")
    return frame


def assign_pool(frame: pd.DataFrame, cutoff_h: int) -> pd.DataFrame:
    """Rows still Needs More Ratings at cutoff_h, with an outcome label.

    ``hours_to_resolution`` is the hours from creation until the note was first
    shown as helpful. It is set only for a positive whose first non-NMR status
    is Helpful. A positive that left as not helpful first has a missing value.
    """
    cutoff_ms = frame["createdAtMillis"] + cutoff_h * HOUR_MS
    first_ts = frame["timestampMillisOfFirstNonNMRStatus"]
    still = first_ts.isna() | (first_ts > cutoff_ms)
    pool = frame.loc[still].copy()

    first_status = pool["firstNonNMRStatus"]
    current = pool["currentStatus"]
    if "latestNonNMRStatus" in pool.columns:
        latest = pool["latestNonNMRStatus"]
    else:
        latest = first_status
    ever_crh = (first_status == CRH) | (latest == CRH) | (current == CRH)
    resolved = pool["timestampMillisOfFirstNonNMRStatus"].notna()

    outcome = pd.Series(OTHER, index=pool.index, dtype=object)
    outcome.loc[~resolved] = UNRESOLVED
    outcome.loc[resolved & (current == CRH)] = POSITIVE
    outcome.loc[resolved & (current != CRH) & ever_crh] = DROPPED
    outcome.loc[resolved & (current == CRNH) & ~ever_crh] = NEGATIVE
    outcome.loc[resolved & (current == NMR) & ~ever_crh] = RETURNED

    hours = (pool["timestampMillisOfFirstNonNMRStatus"] - pool["createdAtMillis"]) / HOUR_MS
    hours = hours.where((outcome == POSITIVE) & (first_status == CRH))

    locked = pd.Series(False, index=pool.index)
    if "lockedStatus" in pool.columns:
        locked = (outcome == POSITIVE) & (pool["lockedStatus"] == CRH)

    return pd.DataFrame(
        {
            "note_id": pool["noteId"].to_numpy(),
            "created_ms": pool["createdAtMillis"].to_numpy(),
            "outcome": outcome.to_numpy(),
            "locked_helpful": locked.to_numpy(),
            "hours_to_resolution": hours.to_numpy(),
        }
    )


def slice_counts(frame: pd.DataFrame, cutoff_h: int) -> dict:
    """Counts for notes that had not yet left NMR at cutoff_h."""
    pool = assign_pool(frame, cutoff_h)
    counts = pool["outcome"].value_counts()

    def n(name: str) -> int:
        return int(counts.get(name, 0))

    positive = n(POSITIVE)
    negative = n(NEGATIVE)
    dropped = n(DROPPED)
    returned = n(RETURNED)
    unresolved = n(UNRESOLVED)
    return {
        "notes": int(len(frame)),
        "still_nmr": int(len(pool)),
        "still_nmr_and_later_resolved": positive + negative + dropped + returned,
        "still_nmr_unresolved_at_snapshot": unresolved,
        "positive_ended_crh": positive,
        "positive_locked_crh": int(pool["locked_helpful"].sum()),
        "negative_never_shown_ended_crnh": negative,
        "dropped_was_shown_not_now": dropped,
        "returned_to_nmr_never_shown": returned,
        "other": n(OTHER),
    }
