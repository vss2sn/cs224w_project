# Community Notes queue (`community_notes_early_graph_triage`)

Rank notes that are still Needs More Ratings at 12 hours. A higher score means show that note first. Publication stays with X's scorer. The positive class is the snapshot status `assign_pool` already writes: the note ends Currently Rated Helpful. That is not "stayed Helpful."

Recall@K is the share of notes the scorer later marks helpful. It is not a measure of usefulness to readers.

## Order

Run from `project/code/` with `PYTHONPATH=.` and the repo venv.

1. `python -m community_notes_early_graph_triage.note_tables` writes `window_notes.parquet` and `pool_12h.parquet` from the status zip. Ids stay strings. Notes created before `HOLDOUT_START` are `train`. Notes from that day are `holdout`. The current dates are in `config.py`.
2. `python -m community_notes_early_graph_triage.stream_ratings` downloads each ratings shard, checks that `createdAtMillis` is a rating clock, and writes `ratings_pre_cutoff_12h.parquet`. Ratings after the note's 12-hour cutoff are dropped. The gate needs 1,000 joins to window notes and a majority strictly later than note creation. It does not judge shard 00000 by itself.
3. `python -m community_notes_early_graph_triage.early_features` writes count and mean rankings to `project/milestone/early_feature_metrics.json`.
4. `python -m community_notes_early_graph_triage.graph` writes `hetero_graph.pt`. Degree counts each rating once, on the rater-to-note edge.
5. `python -m community_notes_early_graph_triage.sample_model` is the smoke test that `train.py` can call a module and return holdout scores. `SampleModel` is not a result row.
6. `python -m community_notes_early_graph_triage.matrix_factorization` fits the simplified rank-1 factorization on train-split ratings only, then scores the holdout by the note intercept with rater terms frozen. Held-out train-split RMSE is a fit check in `rank1_metrics.json`. It is not a holdout ranking metric. Rank 1 is not the publish threshold, and it is not trained as Recall@K.
7. `python -m community_notes_early_graph_triage.encoder` is the submitted encoder run: the same `train.py` call, with `TypedEncoder` swapped in for `SampleModel`. Width 16, one hop, and 40 steps are fixed in `config.py`. That command fits the whole train split. `python -m community_notes_early_graph_triage.encoder --dev` fits notes created before `DEV_START`, keeps the weights with the lowest loss on `[DEV_START, HOLDOUT_START)`, and scores the holdout after that choice. A step count is not chosen from the holdout.

`python -m community_notes_early_graph_triage.run_baselines` is the status-only newest-first and random table. It does not read ratings.

## What the encoder is allowed to see

Rater ids are rows in an embedding table. A note has no free id vector. Its vector is the normalized sum of raters who already rated it. A rater is left out of the sum unless they rated a note the loss uses, and a rater whose fit degree is 0 is left out because dividing by \(\sqrt{0}\) is NaN. That row is not updated on this split. At holdout scoring, rater degree stays the fit degree. Note degree is that note's own pre-cutoff ratings. Scores are joined back to `pool_12h` by `note_id`. Graph order is not pool order.

The loss is binary cross-entropy on train-split notes labeled positive or negative. Dropped notes are an evaluation slice. Unresolved and returned notes are not class 0. Holdout notes are not in the loss, and holdout edges are not in a forward pass that updates a parameter.

An unrated note scores -1 under the mean baseline and sits at intercept 0 under rank 1. Rating-count bin 0 is where those policies disagree.

Recall@K on a tied boundary is the expected count under a uniform draw. NDCG@K with a tie of more than one note is the mean of 32 shuffles from seed 0. Median hours breaks a tie with a seeded hash of the note id. The id encodes creation time, so the smaller id is not used as the order.

## Tests

```bash
cd project/code && PYTHONPATH=. ../../.venv/bin/pytest community_notes_early_graph_triage/test_metrics.py community_notes_early_graph_triage/test_pipeline.py -q
```
