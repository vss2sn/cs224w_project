"""Typed bipartite encoder: one hop, one vector per training rater id.

A note has no free id vector. Its vector is the normalized sum of raters who
have already rated it. A rater is left out of that sum unless they rated a
note the loss uses. A random initial vector is not a trained rater. Dividing
by sqrt(0) is NaN, so a rater whose fit degree is 0 is omitted before the
division. The unknown-rater row is not a trained parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.config import ENCODER_HOPS, ENCODER_STEPS, ENCODER_WIDTH, MILESTONE_DIR, RELATIONS


class TypedEncoder(nn.Module):
    def __init__(self, n_raters: int, width: int = ENCODER_WIDTH, n_features: int = 2):
        super().__init__()
        if width != ENCODER_WIDTH or ENCODER_HOPS != 1:
            raise ValueError("encoder width and hop count are fixed in community_notes_early_graph_triage.config and are not a search")
        self.rater = nn.Embedding(n_raters, width)
        self.alpha = nn.Parameter(torch.empty(len(RELATIONS)))
        # One scale on the aggregated rater vectors. The layer-0 term in the
        # proposal is a free note id, and this model has none, so that term is
        # absent. A coefficient on it would multiply zero and would not train.
        self.beta = nn.Parameter(torch.empty(1))
        self.mlp = nn.Sequential(
            nn.Linear(width + n_features, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        self.initialize()

    def initialize(self) -> None:
        """Replay the seeded init. ``run_training`` calls this after ``set_seed``."""
        self.rater.reset_parameters()
        nn.init.normal_(self.rater.weight, std=0.1)
        with torch.no_grad():
            self.alpha.copy_(torch.tensor([1.0, 0.5, -0.5], device=self.alpha.device, dtype=self.alpha.dtype))
            self.beta.fill_(1)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, data) -> torch.Tensor:
        """One probability per note. Rater degree is `fit_degree`, not recomputed here.

        ``loss_degree`` is the rater's count of ratings on loss notes. A rater
        with none is left out, even if ``fit_degree`` is positive.
        """
        features = data["note"].x
        fit_degree = data["rater"].fit_degree
        loss_degree = data["rater"].loss_degree
        note_degree = data["note"].degree
        width = self.rater.embedding_dim
        aggregated = torch.zeros(features.shape[0], width, device=features.device, dtype=features.dtype)
        for index, relation in enumerate(RELATIONS):
            edge_index = data["rater", relation, "note"].edge_index
            if edge_index.numel() == 0:
                continue
            rater_idx = edge_index[0]
            note_idx = edge_index[1]
            d_u = fit_degree[rater_idx]
            d_n = note_degree[note_idx]
            trained = loss_degree[rater_idx]
            keep = (d_u > 0) & (d_n > 0) & (trained > 0)
            if not bool(keep.any()):
                continue
            coef = self.alpha[index] / torch.sqrt(d_u[keep] * d_n[keep])
            messages = coef.unsqueeze(-1) * self.rater(rater_idx[keep])
            aggregated.index_add_(0, note_idx[keep], messages)
        embedding = self.beta * aggregated
        logit = self.mlp(torch.cat([embedding, features], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


def main() -> None:
    import sys

    import pandas as pd
    import torch as _torch

    from community_notes_early_graph_triage.config import DATA_DIR, SEED
    from community_notes_early_graph_triage.early_features import aggregate_counts, score_frame
    from community_notes_early_graph_triage.metrics import rating_count_bin
    from community_notes_early_graph_triage.train import run_training, set_seed

    use_dev_split = "--dev" in sys.argv
    step_args = [arg for arg in sys.argv[1:] if arg != "--dev"]
    steps = int(step_args[0]) if step_args else ENCODER_STEPS
    report_path = MILESTONE_DIR / "encoder_metrics.json"
    if use_dev_split:
        report_path = MILESTONE_DIR / "encoder_dev_metrics.json"
    elif steps != ENCODER_STEPS:
        report_path = MILESTONE_DIR / f"encoder_{steps}_metrics.json"
    set_seed(SEED)
    blob = _torch.load(DATA_DIR / "hetero_graph.pt", weights_only=False)
    n_raters = len(blob["rater_ids"])
    ratings = pd.read_parquet(DATA_DIR / "ratings_pre_cutoff_12h.parquet", columns=["note_id", "helpfulness"])
    ratings["note_id"] = ratings["note_id"].astype(str)
    pool = pd.read_parquet(DATA_DIR / "pool_12h.parquet")
    pool["note_id"] = pool["note_id"].astype(str)
    holdout = pool.loc[pool["split"] == "holdout"].reset_index(drop=True)
    strata = rating_count_bin(score_frame(holdout, aggregate_counts(ratings))["early_count"].to_numpy())
    run_training(
        TypedEncoder(n_raters),
        steps=steps,
        report_path=report_path,
        pipeline_check=False,
        strata=strata,
        use_dev_split=use_dev_split,
    )


if __name__ == "__main__":
    main()
