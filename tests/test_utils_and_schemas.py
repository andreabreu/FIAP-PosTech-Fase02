"""Tests for seed helpers and interaction schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import Interaction, InteractionBatch
from src.utils import seeded_sample, set_seed


def test_set_seed_is_deterministic() -> None:
    assert set_seed(123) == 123
    assert set_seed(123) == 123


def test_seeded_sample_is_stable() -> None:
    items = list(range(20))
    assert seeded_sample(items, k=5, seed=7) == seeded_sample(items, k=5, seed=7)


def test_interaction_schema_accepts_valid_payload() -> None:
    event = Interaction(user_id=" u1 ", item_id="i9", event_type="purchase", rating=4.5)
    assert event.user_id == "u1"
    assert event.item_id == "i9"


def test_interaction_schema_rejects_blank_ids() -> None:
    with pytest.raises(ValidationError):
        Interaction(user_id=" ", item_id="i1")


def test_interaction_batch_size() -> None:
    batch = InteractionBatch(
        interactions=[
            Interaction(user_id="u1", item_id="i1"),
            Interaction(user_id="u2", item_id="i2"),
        ]
    )
    assert batch.size == 2
