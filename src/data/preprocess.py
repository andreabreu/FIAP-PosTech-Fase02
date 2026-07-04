"""Preprocess raw user-item interactions into a clean training table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.schemas.interactions import Interaction
from src.utils.logging import get_logger

logger = get_logger(__name__)

EVENT_WEIGHT = {
    "view": 1.0,
    "click": 2.0,
    "cart": 3.0,
    "purchase": 5.0,
}


def load_raw_interactions(path: Path) -> pd.DataFrame:
    """Load raw CSV interactions and coerce dtypes.

    Args:
        path: Path to raw interactions CSV.

    Returns:
        pd.DataFrame: Raw frame with normalized columns.
    """
    frame = pd.read_csv(path)
    required = {"user_id", "item_id", "event_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return frame


def validate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows using the Interaction schema.

    Args:
        frame: Raw interactions.

    Returns:
        pd.DataFrame: Validated interactions only.
    """
    records: list[dict] = []
    dropped = 0
    for row in frame.to_dict(orient="records"):
        raw_rating = row.get("rating")
        is_nan_rating = isinstance(raw_rating, float) and pd.isna(raw_rating)
        if raw_rating is None or is_nan_rating:
            rating = None
        elif isinstance(raw_rating, str) and not raw_rating.strip():
            rating = None
        else:
            rating = raw_rating
        raw_ts = row.get("timestamp")
        is_nan_ts = isinstance(raw_ts, float) and pd.isna(raw_ts)
        timestamp = None if raw_ts is None or is_nan_ts else raw_ts
        payload = {
            "user_id": str(row.get("user_id", "")),
            "item_id": str(row.get("item_id", "")),
            "event_type": row.get("event_type", "view"),
            "rating": rating,
            "timestamp": timestamp,
        }
        try:
            records.append(Interaction.model_validate(payload).model_dump())
        except Exception:
            dropped += 1
    logger.info("validated rows=%s dropped=%s", len(records), dropped)
    return pd.DataFrame.from_records(records)


def add_implicit_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive an implicit feedback score from event type / rating.

    Args:
        frame: Validated interactions.

    Returns:
        pd.DataFrame: Frame with ``score`` column.
    """
    out = frame.copy()
    base = out["event_type"].map(EVENT_WEIGHT).fillna(1.0)
    rating = pd.to_numeric(out.get("rating"), errors="coerce")
    out["score"] = base.where(rating.isna(), rating.clip(lower=0.0, upper=5.0))
    return out


def deduplicate_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the strongest score per user-item pair.

    Args:
        frame: Scored interactions.

    Returns:
        pd.DataFrame: Deduplicated interactions.
    """
    sorted_frame = frame.sort_values("score", ascending=False)
    return sorted_frame.drop_duplicates(subset=["user_id", "item_id"], keep="first")


def preprocess_interactions(raw_path: Path, output_path: Path) -> dict[str, int]:
    """Run the full preprocess stage and write a parquet/csv artifact.

    Args:
        raw_path: Raw interactions CSV.
        output_path: Destination path (``.parquet`` or ``.csv``).

    Returns:
        dict[str, int]: Summary counts for logging / DVC metrics.
    """
    raw = load_raw_interactions(raw_path)
    valid = validate_rows(raw)
    scored = add_implicit_score(valid)
    clean = deduplicate_interactions(scored)
    clean = clean.sort_values(["user_id", "item_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        clean.to_parquet(output_path, index=False)
    else:
        clean.to_csv(output_path, index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "clean_rows": int(len(clean)),
        "n_users": int(clean["user_id"].nunique()),
        "n_items": int(clean["item_id"].nunique()),
    }
    logger.info("preprocess summary: %s", summary)
    return summary
