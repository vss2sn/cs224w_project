"""A one-hop linear model that proves train.py can call a module.

There is no rater-id lookup. A learned scalar per edge type is added once per
incoming edge and divided by the note's degree. A degree of 0 skips that
division. This is not the typed encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.config import MILESTONE_DIR, RELATIONS, SMOKE_STEPS


class SampleModel(nn.Module):
    def __init__(self, n_features: int = 2):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.edge_weight = nn.Parameter(torch.empty(len(RELATIONS)))
        self.initialize()

    def initialize(self) -> None:
        """Replay the seeded init. ``run_training`` calls this after ``set_seed``."""
        self.linear.reset_parameters()
        with torch.no_grad():
            self.edge_weight.zero_()

    def forward(self, data) -> torch.Tensor:
        """One probability per note, in the graph's note order."""
        features = data["note"].x
        logit = self.linear(features).squeeze(-1)
        extra = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        degree = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        for index, relation in enumerate(RELATIONS):
            edge_index = data["rater", relation, "note"].edge_index
            if edge_index.numel() == 0:
                continue
            note_idx = edge_index[1]
            extra.index_add_(0, note_idx, self.edge_weight[index].expand(note_idx.shape[0]))
            degree.index_add_(0, note_idx, torch.ones(note_idx.shape[0], device=features.device, dtype=features.dtype))
        # Degree 0 would make the edge term NaN. Those notes are scored from x alone.
        extra = torch.where(degree > 0, extra / degree.clamp(min=1), torch.zeros_like(extra))
        return torch.sigmoid(logit + extra)


def main() -> None:
    import sys

    import pandas as pd

    from community_notes_early_graph_triage.config import DATA_DIR, SEED
    from community_notes_early_graph_triage.early_features import aggregate_counts, score_frame
    from community_notes_early_graph_triage.metrics import rating_count_bin
    from community_notes_early_graph_triage.train import run_training, set_seed

    flags = {"--until-flat", "--dev"}
    until_flat = "--until-flat" in sys.argv
    use_dev_split = "--dev" in sys.argv
    step_args = [arg for arg in sys.argv[1:] if arg not in flags]
    steps = int(step_args[0]) if step_args else SMOKE_STEPS
    report_path = MILESTONE_DIR / "sample_model_metrics.json"
    if use_dev_split:
        report_path = MILESTONE_DIR / "sample_model_dev_metrics.json"
    elif until_flat:
        report_path = MILESTONE_DIR / "sample_model_flat_metrics.json"
    elif steps != SMOKE_STEPS:
        report_path = MILESTONE_DIR / f"sample_model_{steps}_metrics.json"
    set_seed(SEED)
    ratings = pd.read_parquet(DATA_DIR / "ratings_pre_cutoff_12h.parquet", columns=["note_id", "helpfulness"])
    ratings["note_id"] = ratings["note_id"].astype(str)
    pool = pd.read_parquet(DATA_DIR / "pool_12h.parquet")
    pool["note_id"] = pool["note_id"].astype(str)
    holdout = pool.loc[pool["split"] == "holdout"].reset_index(drop=True)
    strata = rating_count_bin(score_frame(holdout, aggregate_counts(ratings))["early_count"].to_numpy())
    run_training(
        SampleModel(),
        steps=steps,
        report_path=report_path,
        pipeline_check=True,
        strata=strata,
        stop_when_flat=until_flat,
        use_dev_split=use_dev_split,
    )


if __name__ == "__main__":
    main()
