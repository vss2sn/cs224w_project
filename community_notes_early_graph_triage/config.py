"""Shared constants for the queue.

Callers import this module. Paths are relative to the repo root, so a
machine-specific absolute path cannot drift between scripts. The study window
is [WINDOW_START, WINDOW_END). Notes created before HOLDOUT_START are the
train split. DEV_START is the optional validation boundary inside that split:
with the dev split on, the fit uses notes created before DEV_START and the
loss that chooses the step uses [DEV_START, HOLDOUT_START). A later window is
a change to these dates. The encoder width, depth, and step cap live here and
are fixed before any holdout metric is computed.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = "2026-11-08"
DATA_DIR = REPO_ROOT / "project" / "data" / "communitynotes" / SNAPSHOT
MILESTONE_DIR = REPO_ROOT / "project" / "milestone"
STATUS_ZIP = DATA_DIR / "noteStatusHistory-00000.zip"

RATINGS_URL = (
    "https://ton.twimg.com/birdwatch-public-data/"
    + SNAPSHOT.replace("-", "/")
    + "/noteRatings/ratings-{shard:05d}.zip"
)

WINDOW_START = "2026-07-01"
WINDOW_END = "2026-10-01"
DEV_START = "2026-08-01"
HOLDOUT_START = "2026-09-01"
CUTOFF_H = 12
HOUR_MS = 60 * 60 * 1000

NOTE_ID_COL = "noteId"
RATER_COL = "raterParticipantId"
CLOCK_COL = "createdAtMillis"
HELPFULNESS_COL = "helpfulnessLevel"

HELPFUL = "HELPFUL"
SOMEWHAT_HELPFUL = "SOMEWHAT_HELPFUL"
NOT_HELPFUL = "NOT_HELPFUL"
HELPFULNESS_CODE = {HELPFUL: 1.0, SOMEWHAT_HELPFUL: 0.5, NOT_HELPFUL: 0.0}

# Live reason tags. The deprecated helpful and notHelpful columns are still
# in the header and are not part of this list.
TAG_COLUMNS = (
    "helpfulOther",
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
    "notHelpfulOther",
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulMissingKeyPoints",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulSpamHarassmentOrAbuse",
    "notHelpfulIrrelevantSources",
    "notHelpfulOpinionSpeculation",
    "notHelpfulNoteNotNeeded",
)

RELATIONS = ("helpful", "somewhat_helpful", "not_helpful")
HELPFULNESS_TO_RELATION = {
    HELPFUL: "helpful",
    SOMEWHAT_HELPFUL: "somewhat_helpful",
    NOT_HELPFUL: "not_helpful",
}

CLOCK_GATE_MIN = 1000

# Frozen before any holdout metric. Not a search.
ENCODER_WIDTH = 16
ENCODER_HOPS = 1
ENCODER_STEPS = 40
SMOKE_STEPS = 20
ADAM_LR = 1e-3
SEED = 0

LAMBDA_I = 0.15
LAMBDA_F = 0.03
MF_MAX_EPOCHS = 20
MF_TOL = 1e-4
MF_HOLDOUT_FRACTION = 0.1
MF_EDGE_BATCH = 262144

NOTE_FEATURES = ("early_count", "early_mean")


def ratings_schema() -> pa.Schema:
    """Stable types for the censored-ratings parquet.

    Declared before the writer opens so a chunk of empty tags cannot change
    a column from integer to float.
    """
    fields = [
        pa.field("note_id", pa.string(), nullable=False),
        pa.field("rater_id", pa.string(), nullable=False),
        pa.field("rating_ms", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("helpfulness", pa.string(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.int64(), nullable=True) for name in TAG_COLUMNS)
    return pa.schema(fields)
