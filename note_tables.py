"""Two parquet tables from the note-status zip.

`window_notes` is every note created in the window, including notes that had
already left Needs More Ratings by 12 hours. `pool_12h` is the queue that
gets ranked. Note ids stay strings; an int/string join would drop every rating.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cn.config import (
    CUTOFF_H,
    DATA_DIR,
    HOLDOUT_START,
    HOUR_MS,
    STATUS_ZIP,
    WINDOW_END,
    WINDOW_START,
)
from cn.labels import assign_pool, load_status


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").value // 10**6)


def window_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Status rows created in [WINDOW_START, WINDOW_END)."""
    start = pd.Timestamp(WINDOW_START, tz="UTC")
    end = pd.Timestamp(WINDOW_END, tz="UTC")
    return frame.loc[(frame["created"] >= start) & (frame["created"] < end)].copy()


def build_window_notes(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per window note: string id, creation time, and the 12-hour cutoff."""
    notes = pd.DataFrame(
        {
            "note_id": frame["noteId"].astype(str),
            "created_ms": frame["createdAtMillis"].astype("int64"),
        }
    )
    notes = notes.drop_duplicates("note_id")
    notes["cutoff_ms"] = notes["created_ms"] + CUTOFF_H * HOUR_MS
    return notes.reset_index(drop=True)


def build_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """The queue at the cutoff, with train before HOLDOUT_START and holdout from that day."""
    pool = assign_pool(frame, CUTOFF_H)
    pool["note_id"] = pool["note_id"].astype(str)
    pool["created_ms"] = pool["created_ms"].astype("int64")
    holdout_ms = _ms(HOLDOUT_START)
    pool["split"] = pd.Series(
        ["train" if created < holdout_ms else "holdout" for created in pool["created_ms"]],
        index=pool.index,
        dtype="object",
    )
    return pool.reset_index(drop=True)


def write_note_tables(status_path: Path, out_dir: Path) -> dict[str, Path]:
    """Write window_notes.parquet and pool_12h.parquet. Returns the two paths."""
    frame = load_status(status_path)
    window = window_frame(frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_path = out_dir / "window_notes.parquet"
    pool_path = out_dir / "pool_12h.parquet"
    build_window_notes(window).to_parquet(notes_path, index=False)
    build_pool(window).to_parquet(pool_path, index=False)
    return {"window_notes": notes_path, "pool_12h": pool_path}


def main() -> None:
    if not STATUS_ZIP.exists():
        raise SystemExit(f"missing {STATUS_ZIP}")
    paths = write_note_tables(STATUS_ZIP, DATA_DIR)
    notes = pd.read_parquet(paths["window_notes"])
    pool = pd.read_parquet(paths["pool_12h"])
    print(f"wrote {paths['window_notes']} rows {len(notes)}")
    print(f"wrote {paths['pool_12h']} rows {len(pool)} splits {pool['split'].value_counts().to_dict()}")
    if not pd.api.types.is_string_dtype(notes["note_id"]):
        raise SystemExit(f"note_id dtype is {notes['note_id'].dtype}, expected string")


if __name__ == "__main__":
    main()
