"""Shared loop for SampleModel and the typed encoder.

The backward pass sees only edges of notes in the fit. Degree in that pass is
the fit degree. Holdout labels are out of the loss, and holdout edges are out
of the graph that pass uses. At scoring time a note's degree is its own
pre-cutoff degree and each rater keeps the frozen fit degree. The fit is every
train note, unless the dev split is on, in which case the fit stops at
DEV_START and the dev interval chooses the step.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.config import ADAM_LR, DATA_DIR, DEV_START, HOLDOUT_START, RELATIONS, SEED
from community_notes_early_graph_triage.graph import filter_edges_to_notes, forward_degrees
from community_notes_early_graph_triage.labels import NEGATIVE, POSITIVE
from community_notes_early_graph_triage.metrics import (
    SCORER_STATUS,
    align_scores,
    code_provenance,
    evaluate_by_stratum,
    evaluate_scores,
    note_tie_break,
    outcome_targets,
    write_json,
)


def set_seed(seed: int) -> None:
    """Restart the RNGs used by embedding init and by note-batch sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

EDGE_BATCH_LIMIT = 4_000_000
NOTE_BATCH = 8192
DEV_PATIENCE = 200
DEV_MIN_DELTA = 1e-4


def loss_tensors(
    pool: pd.DataFrame,
    note_ids: list[str],
    created_from_ms: int | None = None,
    created_before_ms: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph rows whose loss is on. Train split, positive or negative only.

    Dropped, returned, and unresolved notes are not class 0. Holdout notes
    are not in this mask; the edge cut is what keeps their edges out. The
    optional bounds select the fit or the dev interval inside the train split.
    """
    wanted = pool.loc[(pool["split"] == "train") & (pool["outcome"].isin([POSITIVE, NEGATIVE]))]
    if created_from_ms is not None:
        wanted = wanted.loc[wanted["created_ms"] >= created_from_ms]
    if created_before_ms is not None:
        wanted = wanted.loc[wanted["created_ms"] < created_before_ms]
    index = {note_id: i for i, note_id in enumerate(note_ids)}
    rows = []
    labels = []
    for note_id, outcome in zip(wanted["note_id"].astype(str), wanted["outcome"]):
        rows.append(index[note_id])
        labels.append(1.0 if outcome == POSITIVE else 0.0)
    if not rows:
        raise SystemExit("loss mask is empty")
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)


def _edge_count(data) -> int:
    total = 0
    for relation in RELATIONS:
        total += int(data["rater", relation, "note"].edge_index.shape[1])
    return total


def _view_for_notes(data, note_mask: torch.Tensor, fit_degree: torch.Tensor, loss_degree: torch.Tensor):
    view = filter_edges_to_notes(data, note_mask)
    _, note_degree = forward_degrees(view)
    view["rater"].fit_degree = fit_degree
    view["rater"].loss_degree = loss_degree
    view["note"].degree = note_degree
    return view


def _mean_bce(
    model: nn.Module,
    data,
    rows: torch.Tensor,
    label_of: torch.Tensor,
    fit_degree: torch.Tensor,
    loss_degree: torch.Tensor,
    device,
) -> float:
    """Mean binary cross-entropy on `rows`. No step uses the torch RNG."""
    was_training = model.training
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, int(rows.shape[0]), NOTE_BATCH):
            chosen = rows[start : start + NOTE_BATCH]
            mask = torch.zeros(data["note"].num_nodes, dtype=torch.bool)
            mask[chosen.detach().cpu()] = True
            view = _view_for_notes(data, mask, fit_degree, loss_degree).to(device)
            probabilities = model(view)
            loss = torch.nn.functional.binary_cross_entropy(
                probabilities[chosen].clamp(1e-6, 1 - 1e-6), label_of[chosen]
            )
            total += float(loss.detach().cpu()) * int(chosen.shape[0])
            seen += int(chosen.shape[0])
    if was_training:
        model.train()
    if seen == 0:
        raise SystemExit("validation loss is empty")
    return total / seen


def run_training(
    model: nn.Module,
    steps: int,
    report_path: Path,
    pipeline_check: bool,
    strata: np.ndarray | None = None,
    data_dir: Path | None = None,
    stop_when_flat: bool = False,
    use_dev_split: bool = False,
) -> dict:
    """Fit `model` on the train split and score the holdout once.

    The full graph stays on CPU. A training step moves only the edges of the
    notes in that step. Fit degree is computed once, on CPU.

    ``use_dev_split`` fits on notes created before DEV_START and keeps the
    weights with the lowest loss on [DEV_START, HOLDOUT_START). The holdout is
    still scored once, after that choice. The submitted run leaves this off
    and fits the whole train split.
    """
    set_seed(SEED)
    if hasattr(model, "initialize"):
        model.initialize()
    set_seed(SEED)
    data_dir = DATA_DIR if data_dir is None else data_dir
    blob = torch.load(data_dir / "hetero_graph.pt", weights_only=False)
    data = blob["graph"]
    note_ids = blob["note_ids"]
    pool = pd.read_parquet(data_dir / "pool_12h.parquet")
    pool["note_id"] = pool["note_id"].astype(str)
    holdout_ms = int(pd.Timestamp(HOLDOUT_START, tz="UTC").value // 10**6)
    dev_ms = int(pd.Timestamp(DEV_START, tz="UTC").value // 10**6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_mask = data["note"].created_ms < holdout_ms
    # Dev notes stay in the frozen degree when the dev split is off. Turning it
    # on drops those notes from the degree, so the dev interval is not a
    # feature of the fit.
    degree_mask = data["note"].created_ms < dev_ms if use_dev_split else train_mask
    degree_view = filter_edges_to_notes(data, degree_mask)
    fit_degree, _ = forward_degrees(degree_view)
    use_batches = _edge_count(degree_view) > EDGE_BATCH_LIMIT
    del degree_view
    if use_dev_split:
        rows, labels = loss_tensors(pool, note_ids, created_before_ms=dev_ms)
        val_rows, val_labels = loss_tensors(pool, note_ids, created_from_ms=dev_ms, created_before_ms=holdout_ms)
    else:
        rows, labels = loss_tensors(pool, note_ids)
        val_rows, val_labels = None, None
    loss_note_mask = torch.zeros(data["note"].num_nodes, dtype=torch.bool)
    loss_note_mask[rows] = True
    loss_degree, _ = forward_degrees(filter_edges_to_notes(data, loss_note_mask))
    train_view = None if use_batches else _view_for_notes(data, degree_mask, fit_degree, loss_degree).to(device)
    rows = rows.to(device)
    label_of = torch.full((data["note"].num_nodes,), -1.0, device=device)
    label_of[rows] = labels.to(device)
    val_label_of = None
    if val_rows is not None:
        val_rows = val_rows.to(device)
        val_label_of = torch.full((data["note"].num_nodes,), -1.0, device=device)
        val_label_of[val_rows] = val_labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    model.train()
    last_loss = None
    loss_history = []
    dev_loss_history = []
    best_state = None
    best_step = None
    best_dev_loss = None
    since_best = 0
    for step in range(steps):
        optimizer.zero_grad()
        view = train_view
        loss_rows = rows
        if use_batches:
            chosen = rows[torch.randperm(rows.shape[0], device=device)[:NOTE_BATCH]]
            mask = torch.zeros(data["note"].num_nodes, dtype=torch.bool, device=device)
            mask[chosen] = True
            view = _view_for_notes(data, mask.cpu(), fit_degree, loss_degree).to(device)
            loss_rows = chosen
        try:
            probabilities = model(view)
            loss = torch.nn.functional.binary_cross_entropy(
                probabilities[loss_rows].clamp(1e-6, 1 - 1e-6), label_of[loss_rows]
            )
        except RuntimeError as exc:
            if not use_batches and "out of memory" in str(exc).lower():
                use_batches = True
                train_view = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("full batch did not fit; switching to note batches")
                continue
            raise
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        loss_history.append(last_loss)
        if step % 20 == 0 or step == steps - 1:
            print(f"step {step} loss {last_loss:.4f}")
        if stop_when_flat and step >= 100 and step % 20 == 19 and loss_history[-100] - last_loss < 0.001:
            print(f"loss flattened at step {step}")
            break
        if use_dev_split and (step % 20 == 0 or step == steps - 1):
            dev_loss = _mean_bce(model, data, val_rows, val_label_of, fit_degree, loss_degree, device)
            dev_loss_history.append({"step": step, "loss": dev_loss})
            print(f"dev step {step} loss {dev_loss:.4f}")
            improved = best_dev_loss is None or dev_loss < best_dev_loss - DEV_MIN_DELTA
            if improved:
                best_dev_loss = dev_loss
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                since_best = 0
            else:
                since_best += 20
            if since_best >= DEV_PATIENCE:
                print(f"dev loss stopped improving at step {best_step}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    holdout_notes = ~train_mask
    eval_view = _view_for_notes(data, holdout_notes, fit_degree, loss_degree).to(device)
    with torch.no_grad():
        probabilities = model(eval_view).detach().cpu().numpy()
    holdout = pool.loc[pool["split"] == "holdout"].reset_index(drop=True)
    scores = align_scores(np.array(note_ids), probabilities, holdout["note_id"].to_numpy())
    y_positive, y_labeled, hours = outcome_targets(holdout)
    tie_break = note_tie_break(holdout["note_id"])
    saved_steps = (best_step + 1) if best_step is not None else len(loss_history)
    checkpoint = data_dir / f"{report_path.stem}.pt"
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "steps": saved_steps}, checkpoint)
    report = {
        "interpretation": (
            "Pipeline check that train.py returns holdout scores. Not a finding about the queue. "
            if pipeline_check
            else "A gain over the intercept anticipates the scorer's later status. It is not a gain in usefulness. "
        )
        + SCORER_STATUS,
        "steps": saved_steps,
        "steps_ran": len(loss_history),
        "n_loss_notes": int(rows.shape[0]),
        "final_loss": last_loss,
        "loss_history": loss_history,
        "checkpoint": checkpoint.name,
        "provenance": code_provenance(),
        "holdout": evaluate_scores(y_positive, y_labeled, scores, hours, tie_break=tie_break),
    }
    if use_dev_split:
        report["dev_loss_history"] = dev_loss_history
        report["best_dev_step"] = best_step
        report["best_dev_loss"] = best_dev_loss
    if strata is not None:
        report["holdout_by_rating_count"] = evaluate_by_stratum(
            y_positive, y_labeled, scores, hours, strata, tie_break=tie_break
        )
    write_json(report_path, report)
    print(f"wrote {report_path}")
    return report


def main() -> None:
    raise SystemExit("call sample_model.main or encoder.main")


if __name__ == "__main__":
    main()
