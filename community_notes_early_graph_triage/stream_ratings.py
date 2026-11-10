"""Censor the public ratings shards at each note's 12-hour cutoff.

The rating clock is `createdAtMillis`. The public guide describes that column
as the note's creation time, and the file has no second timestamp. The clock
gate joins ratings to window notes across shards until 1,000 matches. Shard
00000 starts in 2021, so that shard alone can match nothing in this window.
Tag columns are kept. `helpful` and `notHelpful` are deprecated and are not
the helpfulness value. `suggestion` is not read: a raw newline in it splits rows.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from community_notes_early_graph_triage.config import (
    CLOCK_COL,
    CLOCK_GATE_MIN,
    DATA_DIR,
    HELPFULNESS_COL,
    HELPFULNESS_CODE,
    NOTE_ID_COL,
    RATER_COL,
    RATINGS_URL,
    TAG_COLUMNS,
    ratings_schema,
)

READ_COLUMNS = (NOTE_ID_COL, RATER_COL, CLOCK_COL, HELPFULNESS_COL, *TAG_COLUMNS)


def clock_gate_decision(n_matched: int, n_later: int, minimum: int = CLOCK_GATE_MIN) -> str:
    """`proceed` only with at least `minimum` joins and a majority strictly later."""
    if n_matched < minimum:
        return "stop_small_sample"
    if n_later * 2 <= n_matched:
        return "stop_not_later"
    return "proceed"


def censor_joined(joined: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep ratings at or before the cutoff and strictly after note creation.

    Rows at or before `created_ms` are clock errors. The returned count is how
    many of those were dropped. Rows after `cutoff_ms` are not yet observed.
    """
    rating_ms = joined["rating_ms"].to_numpy()
    created_ms = joined["created_ms"].to_numpy()
    cutoff_ms = joined["cutoff_ms"].to_numpy()
    clock_error = rating_ms <= created_ms
    keep = (~clock_error) & (rating_ms <= cutoff_ms)
    return joined.loc[keep].copy(), int(clock_error.sum())


def dedup_later(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per rater and note: the later rating."""
    ordered = ratings.sort_values("rating_ms", kind="mergesort")
    return ordered.drop_duplicates(["note_id", "rater_id"], keep="last").reset_index(drop=True)


def dedup_table(table: pa.Table) -> pa.Table:
    """Same rule as dedup_later, without loading the table into pandas."""
    if table.num_rows == 0:
        return table
    ordered = table.sort_by(
        [
            ("note_id", "ascending"),
            ("rater_id", "ascending"),
            ("rating_ms", "ascending"),
        ]
    )
    note = ordered.column("note_id").to_numpy(zero_copy_only=False)
    rater = ordered.column("rater_id").to_numpy(zero_copy_only=False)
    keep = np.empty(ordered.num_rows, dtype=bool)
    keep[-1] = True
    if ordered.num_rows > 1:
        keep[:-1] = (note[:-1] != note[1:]) | (rater[:-1] != rater[1:])
    return ordered.filter(pa.array(keep))


def read_rating_chunks(zip_path: Path):
    """Yield string-typed chunks from a local ratings zip. Ids are not inferred as ints.

    `suggestion` is not a selected column. A raw newline in it would split a row,
    so those rows are skipped and counted. A quoted newline stays inside the field.
    """
    skipped = {"n": 0}

    def handle(_row):
        skipped["n"] += 1
        return "skip"

    parse = pacsv.ParseOptions(delimiter="\t", newlines_in_values=True, invalid_row_handler=handle)
    convert = pacsv.ConvertOptions(
        include_columns=list(READ_COLUMNS),
        column_types={name: pa.string() for name in READ_COLUMNS},
    )
    read = pacsv.ReadOptions(block_size=32 * 1024 * 1024)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".tsv")]
        if len(members) != 1:
            raise SystemExit(f"{zip_path} expected one TSV, found {archive.namelist()}")
        with archive.open(members[0]) as raw:
            reader = pacsv.open_csv(raw, read_options=read, parse_options=parse, convert_options=convert)
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                frame = batch.to_pandas()
                for column in READ_COLUMNS:
                    frame[column] = frame[column].astype("string")
                yield frame, skipped["n"]


def prepare_chunk(chunk: pd.DataFrame, window: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join one chunk to the window and return pre-cutoff rows plus gate counts.

    Gate counts use every joined rating, including those after the cutoff, so
    the column check is not limited to rows the censor keeps.
    """
    frame = chunk.copy()
    frame["note_id"] = frame[NOTE_ID_COL].astype(str)
    frame["rater_id"] = frame[RATER_COL].astype(str)
    frame["rating_ms"] = pd.to_numeric(frame[CLOCK_COL], errors="coerce")
    frame["helpfulness"] = frame[HELPFULNESS_COL].astype(str)
    for column in TAG_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    joined = frame.merge(window, on="note_id", how="inner")
    joined = joined.loc[joined["rating_ms"].notna()]
    later = int((joined["rating_ms"] > joined["created_ms"]).sum())
    known = joined.loc[joined["helpfulness"].isin(HELPFULNESS_CODE)]
    kept, n_clock_errors = censor_joined(known)
    columns = ["note_id", "rater_id", "rating_ms", "helpfulness", *TAG_COLUMNS]
    return kept[columns], {
        "matched": int(len(joined)),
        "strictly_later": later,
        "clock_errors": n_clock_errors,
    }


def _table(frame: pd.DataFrame) -> pa.Table:
    out = frame.copy()
    out["rating_ms"] = pd.to_datetime(out["rating_ms"].astype("int64"), unit="ms", utc=True)
    for column in TAG_COLUMNS:
        out[column] = out[column].astype("Int64")
    return pa.Table.from_pandas(out, schema=ratings_schema(), preserve_index=False)


def should_publish(decision: str, stop_status: int | None) -> bool:
    """Keep a ratings table only after the clock gate and a real end-of-release 404.

    HTTP 403, 429, 500, or running out of the shard cap is a failed download.
    Publishing the rows gathered so far would silently truncate the release.
    """
    return decision == "proceed" and stop_status == 404


def shard_status(url: str) -> int:
    """HTTP status of one ratings zip. A timeout is retried; it is not a 404.

    Some CDNs reject HEAD. A 405 or 501 is checked again with a one-byte GET.
    """
    last_error = None
    status = None
    for attempt in range(3):
        try:
            response = requests.head(url, timeout=90, allow_redirects=True)
            status = int(response.status_code)
            break
        except requests.RequestException as error:
            last_error = error
            print(f"head {url} attempt {attempt + 1} failed: {error}")
    if status is None:
        raise SystemExit(f"head {url} failed after retries: {last_error}")
    if status not in (405, 501):
        return status
    probe = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 60))
    try:
        code = int(probe.status_code)
    finally:
        probe.close()
    if code in (200, 206):
        return 200
    return code


def download_shard(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for piece in response.iter_content(chunk_size=8 * 1024 * 1024):
                if piece:
                    handle.write(piece)


def stream_ratings(window: pd.DataFrame, out_path: Path, tmp_dir: Path) -> dict:
    """Download each shard, censor it, and write one parquet. Deletes the zip after the shard."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()
    writer = None
    matched = 0
    later = 0
    clock_errors = 0
    skipped_lines = 0
    shards = []
    decision = "stop_small_sample"
    gate_message = ""
    stop_status = None
    try:
        for shard in range(100):
            url = RATINGS_URL.format(shard=shard)
            status = shard_status(url)
            if status == 404:
                stop_status = 404
                print(f"shard {shard:05d} 404, stop")
                break
            if status != 200:
                stop_status = status
                gate_message = f"{url} returned HTTP {status}"
                break
            zip_path = tmp_dir / f"ratings-{shard:05d}.zip"
            print(f"downloading {url}")
            try:
                download_shard(url, zip_path)
                seen_skip = 0
                shard_rows = 0
                scanned = 0
                stop_shard = False
                for chunk, skip_total in read_rating_chunks(zip_path):
                    scanned += len(chunk)
                    kept, counts = prepare_chunk(chunk, window)
                    skipped_lines += skip_total - seen_skip
                    seen_skip = skip_total
                    if decision != "proceed":
                        matched += counts["matched"]
                        later += counts["strictly_later"]
                        if matched >= CLOCK_GATE_MIN:
                            decision = clock_gate_decision(matched, later)
                            print(
                                f"clock gate sample {matched} strictly_later {later} decision {decision}"
                            )
                            if decision != "proceed":
                                gate_message = (
                                    f"clock gate {decision}: matched {matched}, strictly_later {later}"
                                )
                                stop_shard = True
                                break
                    clock_errors += counts["clock_errors"]
                    if len(kept):
                        table = _table(kept)
                        if writer is None:
                            writer = pq.ParquetWriter(partial, ratings_schema())
                        writer.write_table(table)
                        shard_rows += len(kept)
                    if scanned % 2_000_000 < len(chunk):
                        print(f"shard {shard:05d} scanned {scanned} matched {matched} kept {shard_rows}")
                shards.append({"shard": shard, "kept_rows": shard_rows})
                print(f"shard {shard:05d} kept {shard_rows}")
                if stop_shard:
                    break
            finally:
                if zip_path.exists():
                    zip_path.unlink()
        else:
            gate_message = "stopped after 100 shards without a 404"
    finally:
        if writer is not None:
            writer.close()
    if not should_publish(decision, stop_status):
        if partial.exists():
            partial.unlink()
        if decision != "proceed" and not gate_message:
            gate_message = f"clock gate {decision}: matched {matched}, strictly_later {later}"
        if not gate_message:
            gate_message = f"ratings download stopped at HTTP {stop_status}, not 404"
        raise SystemExit(gate_message)
    if not partial.exists():
        raise SystemExit("clock gate passed but no pre-cutoff ratings were kept")
    table = dedup_table(pq.read_table(partial))
    temporary = out_path.with_name(out_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, ratings_schema())
    writer.write_table(table)
    writer.close()
    os.replace(temporary, out_path)
    partial.unlink()
    report = {
        "shards": shards,
        "matched_for_gate": matched,
        "strictly_later": later,
        "clock_gate": decision,
        "clock_errors_dropped": clock_errors,
        "malformed_lines_skipped": skipped_lines,
        "rows": int(table.num_rows),
    }
    return report


def _millis(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values, utc=True)
    millis = stamps.astype("int64")
    if len(millis) and int(millis.max()) > 10**14:
        millis = millis // 10**6
    return millis.astype("int64")


def main() -> None:
    from community_notes_early_graph_triage.metrics import write_json
    from community_notes_early_graph_triage.config import MILESTONE_DIR

    window_path = DATA_DIR / "window_notes.parquet"
    if not window_path.exists():
        raise SystemExit(f"missing {window_path}")
    window = pd.read_parquet(window_path)
    window["note_id"] = window["note_id"].astype(str)
    out_path = DATA_DIR / "ratings_pre_cutoff_12h.parquet"
    report = stream_ratings(window[["note_id", "created_ms", "cutoff_ms"]], out_path, DATA_DIR)
    write_json(MILESTONE_DIR / "ratings_stream.json", report)
    print(f"wrote {out_path} rows {report['rows']} gate {report['clock_gate']}")


if __name__ == "__main__":
    main()
