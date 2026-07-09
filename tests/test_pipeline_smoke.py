"""Smoke tests for preprocess and feature engineering stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.data.preprocess import preprocess_interactions
from src.features.engineering import build_features


def test_preprocess_and_features(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    processed = tmp_path / "clean.csv"
    features = tmp_path / "features"
    params = tmp_path / "params.yaml"
    params.write_text(
        "seed: 0\nfeature_eng:\n  test_ratio: 0.25\n  min_user_interactions: 1\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "user_id": "U1",
                "item_id": "I1",
                "event_type": "view",
                "rating": "",
                "timestamp": "2025-01-01T00:00:00",
            },
            {
                "user_id": "U1",
                "item_id": "I2",
                "event_type": "purchase",
                "rating": "4.5",
                "timestamp": "2025-01-02T00:00:00",
            },
            {
                "user_id": "U2",
                "item_id": "I1",
                "event_type": "click",
                "rating": "",
                "timestamp": "2025-01-03T00:00:00",
            },
        ]
    ).to_csv(raw, index=False)

    summary = preprocess_interactions(raw, processed)
    assert summary["clean_rows"] == 3
    assert processed.exists()

    feat_summary = build_features(processed, features, params)
    assert feat_summary["train_rows"] >= 1
    assert (features / "train.csv").exists()
    assert (features / "id_maps.json").exists()
