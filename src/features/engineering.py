"""User-item feature engineering for recommender training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_params(path: Path) -> dict:
    """Load pipeline parameters from ``params.yaml``.

    Args:
        path: Path to params file.

    Returns:
        dict: Nested parameter mapping.
    """
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def encode_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Map string user/item ids to contiguous integer indices.

    Args:
        frame: Clean interactions with ``user_id`` / ``item_id``.

    Returns:
        tuple: Encoded frame and id maps.
    """
    out = frame.copy()
    user_ids = sorted(out["user_id"].astype(str).unique())
    item_ids = sorted(out["item_id"].astype(str).unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}
    out["user_idx"] = out["user_id"].astype(str).map(user_map)
    out["item_idx"] = out["item_id"].astype(str).map(item_map)
    return out, {"user_map": user_map, "item_map": item_map}


def train_test_split_by_user(
    frame: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of each user's interactions for evaluation.

    Args:
        frame: Encoded interactions.
        test_ratio: Fraction of interactions per user reserved for test.
        seed: RNG seed.

    Returns:
        tuple: ``(train_df, test_df)``.
    """
    parts_train: list[pd.DataFrame] = []
    parts_test: list[pd.DataFrame] = []
    shuffled = frame.sample(frac=1.0, random_state=seed)
    for _, group in shuffled.groupby("user_idx", sort=False):
        n = len(group)
        n_test = max(1, int(round(n * test_ratio))) if n > 1 else 0
        if n_test == 0:
            parts_train.append(group)
            continue
        test = group.iloc[:n_test]
        train = group.iloc[n_test:]
        if train.empty:
            parts_train.append(group)
        else:
            parts_train.append(train)
            parts_test.append(test)
    train_df = pd.concat(parts_train, ignore_index=True) if parts_train else frame
    test_df = (
        pd.concat(parts_test, ignore_index=True) if parts_test else frame.iloc[0:0]
    )
    return train_df, test_df


def build_features(
    processed_path: Path,
    output_dir: Path,
    params_path: Path,
) -> dict[str, int]:
    """Build encoded train/test feature tables for downstream training.

    Args:
        processed_path: Clean interactions from preprocess stage.
        output_dir: Directory for feature artifacts.
        params_path: Path to ``params.yaml``.

    Returns:
        dict[str, int]: Summary counts.
    """
    params = load_params(params_path)
    feature_params = params.get("feature_eng", {})
    test_ratio = float(feature_params.get("test_ratio", 0.2))
    seed = int(feature_params.get("seed", params.get("seed", 42)))
    min_interactions = int(feature_params.get("min_user_interactions", 2))

    if processed_path.suffix == ".parquet":
        frame = pd.read_parquet(processed_path)
    else:
        frame = pd.read_csv(processed_path)

    counts = frame.groupby("user_id").size()
    keep_users = counts[counts >= min_interactions].index
    filtered = frame[frame["user_id"].isin(keep_users)].copy()
    encoded, maps = encode_ids(filtered)
    train_df, test_df = train_test_split_by_user(
        encoded,
        test_ratio=test_ratio,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    maps_path = output_dir / "id_maps.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    serializable = {
        "user_map": maps["user_map"],
        "item_map": maps["item_map"],
        "n_users": len(maps["user_map"]),
        "n_items": len(maps["item_map"]),
    }
    maps_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    summary = {
        "input_rows": int(len(frame)),
        "filtered_rows": int(len(filtered)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_users": int(len(maps["user_map"])),
        "n_items": int(len(maps["item_map"])),
    }
    logger.info("feature_eng summary: %s", summary)
    return summary
