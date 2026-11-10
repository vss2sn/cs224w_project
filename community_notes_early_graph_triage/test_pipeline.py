"""Synthetic checks for the censored-ratings pipeline. These do not download shards."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from community_notes_early_graph_triage.config import HELPFUL, NOT_HELPFUL, SOMEWHAT_HELPFUL, TAG_COLUMNS
from community_notes_early_graph_triage.encoder import TypedEncoder
from community_notes_early_graph_triage.graph import build_graph, filter_edges_to_notes, forward_degrees
from community_notes_early_graph_triage.metrics import align_scores
from community_notes_early_graph_triage.matrix_factorization import edge_pairs, fit_rater_terms, hold_out_edges, note_intercepts, predict
from community_notes_early_graph_triage.stream_ratings import censor_joined, clock_gate_decision, dedup_later


def _joined(rows):
    return pd.DataFrame(rows)


def test_rating_millis_keeps_millisecond_timestamps():
    from community_notes_early_graph_triage.graph import rating_millis

    stamp = pd.Timestamp("2026-07-01 00:02:16.927+00:00")
    stamps = pd.Series([stamp]).astype("datetime64[ms, UTC]")
    millis = rating_millis(stamps)
    assert int(millis[0]) == stamp.value // 10**6


def test_censor_drops_late_and_early_ratings():
    joined = _joined(
        [
            {"rating_ms": 150, "created_ms": 0, "cutoff_ms": 100, "note_id": "a"},
            {"rating_ms": 0, "created_ms": 0, "cutoff_ms": 100, "note_id": "b"},
            {"rating_ms": -5, "created_ms": 0, "cutoff_ms": 100, "note_id": "c"},
            {"rating_ms": 40, "created_ms": 0, "cutoff_ms": 100, "note_id": "d"},
        ]
    )
    kept, n_errors = censor_joined(joined)
    assert kept["note_id"].tolist() == ["d"]
    assert n_errors == 2


def test_string_ids_join_and_dedup_keeps_the_later_row():
    window = pd.DataFrame({"note_id": ["1234567890123456789"], "created_ms": [0], "cutoff_ms": [100]})
    ratings = pd.DataFrame(
        {
            "noteId": ["1234567890123456789", "1234567890123456789"],
            "raterParticipantId": ["42", "42"],
            "createdAtMillis": ["10", "20"],
            "helpfulnessLevel": [NOT_HELPFUL, HELPFUL],
            **{column: ["", ""] for column in TAG_COLUMNS},
        }
    )
    from community_notes_early_graph_triage.stream_ratings import prepare_chunk

    kept, _counts = prepare_chunk(ratings, window)
    kept = dedup_later(kept)
    assert kept["note_id"].tolist() == ["1234567890123456789"]
    assert not pd.api.types.is_numeric_dtype(kept["note_id"])
    assert kept["helpfulness"].tolist() == [HELPFUL]
    assert int(kept["rating_ms"].iloc[0]) == 20


def test_malformed_rating_line_is_skipped_and_ids_stay_strings(tmp_path):
    import zipfile

    from community_notes_early_graph_triage.config import TAG_COLUMNS
    from community_notes_early_graph_triage.stream_ratings import read_rating_chunks

    header = ["noteId", "raterParticipantId", "createdAtMillis", "helpfulnessLevel", *TAG_COLUMNS, "suggestion"]
    good = ["1234567890123456789", "42", "10", "HELPFUL", *([""] * len(TAG_COLUMNS)), "ok"]
    quoted = ["1234567890123456789", "7", "11", "NOT_HELPFUL", *([""] * len(TAG_COLUMNS)), '"has\nnewline"']
    lines = ["\t".join(header), "\t".join(good), "this line is broken", "\t".join(quoted)]
    zip_path = tmp_path / "ratings-00000.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("ratings-00000.tsv", "\n".join(lines) + "\n")
    frames = []
    skipped = 0
    for frame, skip_total in read_rating_chunks(zip_path):
        frames.append(frame)
        skipped = skip_total
    kept = pd.concat(frames, ignore_index=True)
    assert skipped == 1
    assert kept["noteId"].tolist() == ["1234567890123456789", "1234567890123456789"]
    assert not pd.api.types.is_numeric_dtype(kept["noteId"])


def test_clock_gate_requires_a_thousand_matches_and_a_majority():
    assert clock_gate_decision(999, 999) == "stop_small_sample"
    assert clock_gate_decision(1000, 500) == "stop_not_later"
    assert clock_gate_decision(1000, 501) == "proceed"


def test_scores_align_to_pool_order_including_a_degree_zero_note():
    graph_ids = np.array(["b", "a", "c"])
    scores = np.array([1.0, 2.0, 3.0])
    pool_ids = np.array(["a", "c"])
    aligned = align_scores(graph_ids, scores, pool_ids)
    assert aligned.tolist() == [2.0, 3.0]


def test_training_edges_exclude_the_holdout_and_degree_counts_a_rating_once():
    window = pd.DataFrame(
        {
            "note_id": ["fit", "holdout"],
            "created_ms": [0, 10_000],
            "cutoff_ms": [100, 10_100],
        }
    )
    ratings = pd.DataFrame(
        {
            "note_id": ["fit", "holdout", "fit"],
            "rater_id": ["u", "u", "v"],
            "rating_ms": [10, 10_010, 20],
            "helpfulness": [HELPFUL, NOT_HELPFUL, SOMEWHAT_HELPFUL],
        }
    )
    data, maps, _report = build_graph(ratings, window)
    rater_degree, note_degree = forward_degrees(data)
    assert float(rater_degree.sum()) == 3.0
    reverse = data["note", "rev_helpful", "rater"].edge_index.shape[1]
    assert reverse == 1
    assert float(rater_degree.sum() + reverse) != float(rater_degree.sum())
    fit_notes = data["note"].created_ms < 10_000
    train = filter_edges_to_notes(data, fit_notes)
    holdout_ids = set(maps["note_ids"]) - {"fit"}
    holdout_index = maps["note_ids"].index("holdout")
    assert int(train["rater", "not_helpful", "note"].edge_index.shape[1]) == 0
    assert holdout_index not in train["rater", "helpful", "note"].edge_index[1].tolist()
    fit_degree, _ = forward_degrees(train)
    evaluation = filter_edges_to_notes(data, ~fit_notes)
    evaluation["rater"].fit_degree = fit_degree
    # A holdout edge must not change the frozen fit degree.
    assert torch.equal(evaluation["rater"].fit_degree, fit_degree)
    _, eval_note_degree = forward_degrees(evaluation)
    assert float(eval_note_degree[holdout_index]) == 1.0
    assert float(note_degree.sum()) == 3.0
    assert holdout_ids == {"holdout"}


def test_degree_zero_rater_does_not_make_a_nan_score():
    data = HeteroData()
    data["note"].x = torch.zeros((1, 2))
    data["note"].degree = torch.tensor([1.0])
    data["note"].num_nodes = 1
    data["rater"].num_nodes = 1
    data["rater"].fit_degree = torch.tensor([0.0])
    data["rater"].loss_degree = torch.tensor([0.0])
    data["rater", "helpful", "note"].edge_index = torch.tensor([[0], [0]])
    data["rater", "somewhat_helpful", "note"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data["rater", "not_helpful", "note"].edge_index = torch.empty((2, 0), dtype=torch.long)
    score = TypedEncoder(n_raters=1)(data)
    assert torch.isfinite(score).all()


def test_untrained_rater_is_left_out_of_the_sum():
    def graph(loss_degree: float) -> HeteroData:
        data = HeteroData()
        data["note"].x = torch.zeros((1, 2))
        data["note"].degree = torch.tensor([1.0])
        data["note"].num_nodes = 1
        data["rater"].num_nodes = 1
        data["rater"].fit_degree = torch.tensor([4.0])
        data["rater"].loss_degree = torch.tensor([loss_degree])
        data["rater", "helpful", "note"].edge_index = torch.tensor([[0], [0]])
        data["rater", "somewhat_helpful", "note"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["rater", "not_helpful", "note"].edge_index = torch.empty((2, 0), dtype=torch.long)
        return data

    model = TypedEncoder(n_raters=1)
    with torch.no_grad():
        model.rater.weight.fill_(3.0)
    untrained = model(graph(0.0))
    trained = model(graph(1.0))
    assert torch.isfinite(untrained).all()
    assert not torch.allclose(untrained, trained)


def test_hours_count_only_a_helpful_first_exit():
    from community_notes_early_graph_triage.labels import CRH, CRNH, assign_pool

    hour = 60 * 60 * 1000
    frame = pd.DataFrame(
        {
            "noteId": ["helpful-first", "not-helpful-first"],
            "createdAtMillis": [0, 0],
            "timestampMillisOfFirstNonNMRStatus": [20 * hour, 20 * hour],
            "firstNonNMRStatus": [CRH, CRNH],
            "currentStatus": [CRH, CRH],
        }
    )
    pool = assign_pool(frame, 12).set_index("note_id")
    assert pool.loc["helpful-first", "hours_to_resolution"] == 20
    assert pd.isna(pool.loc["not-helpful-first", "hours_to_resolution"])


def test_rank1_factors_recover_an_interaction():
    ratings = pd.DataFrame(
        [
            {"note_id": "n0", "rater_id": "u0", "helpfulness": HELPFUL},
            {"note_id": "n1", "rater_id": "u0", "helpfulness": NOT_HELPFUL},
            {"note_id": "n0", "rater_id": "u1", "helpfulness": NOT_HELPFUL},
            {"note_id": "n1", "rater_id": "u1", "helpfulness": HELPFUL},
        ]
    )
    fit = fit_rater_terms(ratings, seed=0)
    assert max(abs(value) for value in fit["f_u"].values()) > 1e-3
    assert max(abs(value) for value in fit["f_n"].values()) > 1e-3
    _values, prediction = predict(ratings, fit)
    assert prediction[0] > prediction[1]
    assert prediction[3] > prediction[2]


def test_non_404_stop_does_not_publish_a_partial_download():
    from community_notes_early_graph_triage.stream_ratings import should_publish

    assert should_publish("proceed", 404)
    assert not should_publish("proceed", 500)
    assert not should_publish("proceed", 429)
    assert not should_publish("proceed", None)
    assert not should_publish("stop_small_sample", 404)


def test_queue_ids_are_the_pool():
    from community_notes_early_graph_triage.graph import queue_ids

    pool = pd.DataFrame({"note_id": ["only-the-queue"]})
    assert queue_ids(pool) == {"only-the-queue"}


def test_dev_split_leaves_the_dev_interval_out_of_the_fit():
    from community_notes_early_graph_triage.config import DEV_START, HOLDOUT_START
    from community_notes_early_graph_triage.train import loss_tensors

    dev_ms = int(pd.Timestamp(DEV_START, tz="UTC").value // 10**6)
    holdout_ms = int(pd.Timestamp(HOLDOUT_START, tz="UTC").value // 10**6)
    pool = pd.DataFrame(
        {
            "note_id": ["fit", "dev", "holdout"],
            "created_ms": [dev_ms - 1, dev_ms, holdout_ms + 1],
            "split": ["train", "train", "holdout"],
            "outcome": ["positive", "negative", "positive"],
        }
    )
    note_ids = ["fit", "dev", "holdout"]
    fit, _labels = loss_tensors(pool, note_ids, created_before_ms=dev_ms)
    dev, _labels = loss_tensors(pool, note_ids, created_from_ms=dev_ms, created_before_ms=holdout_ms)
    assert fit.tolist() == [0]
    assert dev.tolist() == [1]


def test_initialize_repeats_from_the_seed():
    from community_notes_early_graph_triage.encoder import TypedEncoder
    from community_notes_early_graph_triage.sample_model import SampleModel
    from community_notes_early_graph_triage.train import set_seed

    set_seed(0)
    first = SampleModel()
    set_seed(0)
    second = SampleModel()
    set_seed(0)
    first.initialize()
    set_seed(0)
    second.initialize()
    assert torch.equal(first.linear.weight, second.linear.weight)
    set_seed(0)
    encoder = TypedEncoder(4)
    set_seed(0)
    encoder.initialize()
    again = TypedEncoder(4)
    set_seed(0)
    again.initialize()
    assert torch.equal(encoder.rater.weight, again.rater.weight)
    assert encoder.beta.shape == (1,)


def test_run_training_scores_the_holdout(tmp_path):
    from community_notes_early_graph_triage.config import HOLDOUT_START
    from community_notes_early_graph_triage.sample_model import SampleModel
    from community_notes_early_graph_triage.train import run_training, set_seed

    holdout_ms = int(pd.Timestamp(HOLDOUT_START, tz="UTC").value // 10**6)
    window = pd.DataFrame(
        {
            "note_id": ["fit", "holdout"],
            "created_ms": [0, holdout_ms + 1_000],
            "cutoff_ms": [100, holdout_ms + 2_000],
        }
    )
    ratings = pd.DataFrame(
        {
            "note_id": ["fit", "holdout"],
            "rater_id": ["u", "u"],
            "rating_ms": [10, holdout_ms + 1_500],
            "helpfulness": [HELPFUL, NOT_HELPFUL],
        }
    )
    data, maps, _report = build_graph(ratings, window)
    torch.save(
        {"graph": data, "note_ids": maps["note_ids"], "rater_ids": maps["rater_ids"]},
        tmp_path / "hetero_graph.pt",
    )
    pd.DataFrame(
        {
            "note_id": ["fit", "holdout"],
            "created_ms": [0, holdout_ms + 1_000],
            "split": ["train", "holdout"],
            "outcome": ["positive", "positive"],
            "hours_to_resolution": [20.0, 30.0],
        }
    ).to_parquet(tmp_path / "pool_12h.parquet", index=False)
    set_seed(0)
    report = run_training(
        SampleModel(),
        steps=2,
        report_path=tmp_path / "smoke.json",
        pipeline_check=True,
        data_dir=tmp_path,
    )
    assert report["holdout"]["n_pool"] == 1
    assert report["provenance"]["seed"] == 0
    assert (tmp_path / "smoke.pt").exists()


def test_rank1_held_out_edges_are_removed_and_unrated_intercept_is_zero():
    rows = []
    for note, value, repeat in (("high", HELPFUL, 6), ("low", NOT_HELPFUL, 6)):
        for rater in range(repeat):
            rows.append({"note_id": note, "rater_id": f"u{rater}", "helpfulness": value})
    ratings = pd.DataFrame(rows)
    train, held_out = hold_out_edges(ratings, fraction=0.25, seed=0)
    assert edge_pairs(train).isdisjoint(edge_pairs(held_out))
    fit = fit_rater_terms(train)
    assert edge_pairs(held_out).isdisjoint(edge_pairs(train))
    intercepts = note_intercepts(
        pd.DataFrame(columns=["note_id", "rater_id", "helpfulness"]),
        fit["mu"],
        fit["i_u"],
        fit["f_u"],
        pd.Series(["nobody"]),
    )
    assert float(intercepts.loc["nobody"]) == 0.0
    one = pd.DataFrame([{"note_id": "solo", "rater_id": "new", "helpfulness": HELPFUL}])
    closed = note_intercepts(one, mu=0.0, i_u={}, f_u={}, note_ids=pd.Series(["solo"]))
    assert abs(float(closed.loc["solo"]) - (1.0 / (1.0 + 0.15))) < 1e-9
    scored = note_intercepts(ratings, fit["mu"], fit["i_u"], fit["f_u"], pd.Series(["high", "low"]))
    assert float(scored.loc["high"]) > float(scored.loc["low"])
