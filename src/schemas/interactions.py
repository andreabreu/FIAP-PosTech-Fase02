"""Schemas for user-item interaction events."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Interaction(BaseModel):
    """Single user-item interaction event."""

    user_id: str = Field(..., min_length=1, description="User identifier")
    item_id: str = Field(..., min_length=1, description="Item identifier")
    event_type: Literal["view", "click", "cart", "purchase"] = "view"
    rating: float | None = Field(default=None, ge=0.0, le=5.0)
    timestamp: datetime | None = None

    @field_validator("user_id", "item_id")
    @classmethod
    def strip_ids(cls, value: str) -> str:
        """Normalize identifiers by stripping whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("identifier must not be blank")
        return cleaned


class InteractionBatch(BaseModel):
    """Batch of interactions used for training or evaluation."""

    interactions: list[Interaction] = Field(default_factory=list)

    @property
    def size(self) -> int:
        """Return the number of interactions in the batch."""
        return len(self.interactions)
