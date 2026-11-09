"""Typed rater–note graph of pre-cutoff ratings.

Every edge must fall at or before that note's cutoff. Degree counts a rating
once, on the rater-to-note edge. The reverse edge is the same rating stored
the other way so a later layer can update both sides. Note features are early
count and early mean, and the mean is 0 when the count is 0. Age is not a
feature: every note in this queue is scored at 12 hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cn.config import (
    DATA_DIR,
    HELPFULNESS_TO_RELATION,
    MILESTONE_DIR,
    RELATIONS,
)
from cn.early_features import aggregate_counts
from cn.metrics import write_json


def _store(data: HeteroData, edge_type, src: np.ndarray, dst: np.ndarray, rating_ms: np.ndarray) -> None:
    store = data[edge_type]
    if len(src) == 0:
        store.edge_index = torch.empty((2, 0), dtype=torch.long)
        store.rating_ms = torch.empty((0,), dtype=torch.long)
        return
    store.edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    store.rating_ms = torch.tensor(rating_ms, dtype=torch.long)


def queue_ids(pool: pd.DataFrame) -> set[str]:
    """Ids of the ranked queue. This is not every note in the window."""
    return set(pool["note_id"].astype(str))


def filter_edges_to_notes(data: HeteroData, note_mask: torch.Tensor) -> HeteroData:
    """Edges that touch only notes where `note_mask` is true.

    Node indices are unchanged. The copy stores those edges only, not a clone
    of every edge in `data`. A holdout edge is removed by masking out
    holdout notes; it is not hidden by a loss weight.
    """
    out = HeteroData()
    out["note"].x = data["note"].x
    out["note"].created_ms = data["note"].created_ms
    out["note"].num_nodes = data["note"].num_nodes
    out["rater"].num_nodes = data["rater"].num_nodes
    for relation in RELATIONS:
        _copy_kept(out, data, ("rater", relation, "note"), note_mask, endpoint=1)
        _copy_kept(out, data, ("note", f"rev_{relation}", "rater"), note_mask, endpoint=0)
    return out


def _copy_kept(out: HeteroData, data: HeteroData, edge_type, note_mask: torch.Tensor, endpoint: int) -> None:
    source = data[edge_type]
    edge_index = source.edge_index
    rating_ms = source.rating_ms
    if edge_index.numel() == 0:
        out[edge_type].edge_index = edge_index
        out[edge_type].rating_ms = rating_ms
        return
    keep = note_mask[edge_index[endpoint]]
    out[edge_type].edge_index = edge_index[:, keep]
    out[edge_type].rating_ms = rating_ms[keep]


def forward_degrees(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
    """Degree from rater-to-note edges only. Reverse edges are not a second rating."""
    n_raters = int(data["rater"].num_nodes)
    n_notes = int(data["note"].num_nodes)
    device = data["note"].x.device
    rater = torch.zeros(n_raters, dtype=torch.float32, device=device)
    note = torch.zeros(n_notes, dtype=torch.float32, device=device)
    for relation in RELATIONS:
        edge_index = data["rater", relation, "note"].edge_index
        if edge_index.numel() == 0:
            continue
        ones = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
        rater.scatter_add_(0, edge_index[0], ones)
        note.scatter_add_(0, edge_index[1], ones)
    return rater, note


def rating_millis(values: pd.Series) -> np.ndarray:
    """Unix milliseconds. Datetime64[ms] is already milliseconds; datetime64[ns] is not."""
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=np.int64)
    millis = pd.to_datetime(values, utc=True).astype("int64").to_numpy()
    if len(millis) and int(millis.max()) > 10**14:
        millis = millis // 10**6
    return millis.astype(np.int64)


def build_graph(ratings: pd.DataFrame, window: pd.DataFrame) -> tuple[HeteroData, dict, dict]:
    """HeteroData plus id maps and a short degree report.

    Notes with no ratings stay in the graph at degree 0. Their feature mean is 0.
    """
    notes = window.copy()
    notes["note_id"] = notes["note_id"].astype(str)
    notes = notes.drop_duplicates("note_id").reset_index(drop=True)
    note_ids = notes["note_id"].tolist()
    note_index = {note_id: i for i, note_id in enumerate(note_ids)}

    frame = ratings.copy()
    frame["note_id"] = frame["note_id"].astype(str)
    frame["rater_id"] = frame["rater_id"].astype(str)
    frame["rating_ms"] = rating_millis(frame["rating_ms"])
    frame = frame.loc[frame["note_id"].isin(note_index)]
    cutoff = notes.set_index("note_id")["cutoff_ms"]
    frame["cutoff_ms"] = frame["note_id"].map(cutoff).astype("int64")
    if len(frame) and bool((frame["rating_ms"] > frame["cutoff_ms"]).any()):
        raise AssertionError("a rating is later than its note's cutoff")

    rater_ids = sorted(frame["rater_id"].unique().tolist())
    rater_index = {rater_id: i for i, rater_id in enumerate(rater_ids)}
    counts = aggregate_counts(frame) if len(frame) else pd.DataFrame(columns=["note_id", "early_count", "early_mean"])
    features = notes[["note_id"]].merge(counts, on="note_id", how="left")
    early_count = features["early_count"].fillna(0).to_numpy(dtype=np.float32)
    early_mean = features["early_mean"].fillna(0).to_numpy(dtype=np.float32)
    # fillna(0), not -1: the mean baseline's missing-value score is a different policy.
    x = np.column_stack([early_count, early_mean])

    data = HeteroData()
    data["note"].x = torch.tensor(x, dtype=torch.float32)
    data["note"].created_ms = torch.tensor(notes["created_ms"].to_numpy(dtype=np.int64), dtype=torch.long)
    data["note"].num_nodes = len(note_ids)
    data["rater"].num_nodes = len(rater_ids)

    edges_per_type = {}
    for relation in RELATIONS:
        _store(data, ("rater", relation, "note"), np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        _store(data, ("note", f"rev_{relation}", "rater"), np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    if len(frame):
        frame["note_idx"] = frame["note_id"].map(note_index).astype(np.int64)
        frame["rater_idx"] = frame["rater_id"].map(rater_index).astype(np.int64)
        frame["relation"] = frame["helpfulness"].map(HELPFULNESS_TO_RELATION)
        unknown = frame["relation"].isna()
        if bool(unknown.any()):
            raise ValueError(f"unknown helpfulness values: {sorted(frame.loc[unknown, 'helpfulness'].unique())}")
        for relation, group in frame.groupby("relation", sort=False):
            src = group["rater_idx"].to_numpy()
            dst = group["note_idx"].to_numpy()
            when = group["rating_ms"].to_numpy()
            _store(data, ("rater", relation, "note"), src, dst, when)
            _store(data, ("note", f"rev_{relation}", "rater"), dst, src, when)
            edges_per_type[str(relation)] = int(len(group))

    rater_degree, note_degree = forward_degrees(data)
    report = {
        "n_raters": len(rater_ids),
        "n_notes": len(note_ids),
        "edges_per_type": edges_per_type,
        "notes_degree_0": int((note_degree == 0).sum()),
        "pool_notes_degree_0": None,
    }
    maps = {"note_ids": note_ids, "rater_ids": rater_ids}
    return data, maps, report


def pool_degree_zero(note_ids: list[str], note_degree: torch.Tensor, pool_note_ids: pd.Series) -> int:
    index = {note_id: i for i, note_id in enumerate(note_ids)}
    degrees = np.array([float(note_degree[index[note_id]]) for note_id in pool_note_ids.astype(str)])
    return int((degrees == 0).sum())


def write_graph(ratings: pd.DataFrame, window: pd.DataFrame, pool: pd.DataFrame, out_dir: Path) -> dict:
    data, maps, report = build_graph(ratings, window)
    maps["pool_ids"] = queue_ids(pool)
    _, note_degree = forward_degrees(data)
    report["pool_notes_degree_0"] = pool_degree_zero(maps["note_ids"], note_degree, pool["note_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"graph": data, "note_ids": maps["note_ids"], "rater_ids": maps["rater_ids"]}, out_dir / "hetero_graph.pt")
    pd.DataFrame({"note_id": maps["note_ids"]}).to_parquet(out_dir / "note_id_map.parquet", index=False)
    pd.DataFrame({"rater_id": maps["rater_ids"]}).to_parquet(out_dir / "rater_id_map.parquet", index=False)
    return report


def main() -> None:
    ratings = pd.read_parquet(DATA_DIR / "ratings_pre_cutoff_12h.parquet")
    window = pd.read_parquet(DATA_DIR / "window_notes.parquet")
    pool = pd.read_parquet(DATA_DIR / "pool_12h.parquet")
    report = write_graph(ratings, window, pool, DATA_DIR)
    write_json(MILESTONE_DIR / "graph_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
