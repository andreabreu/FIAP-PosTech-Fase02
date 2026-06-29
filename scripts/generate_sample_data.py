#!/usr/bin/env python3
"""Generate a synthetic user-item interactions CSV for local/DVC demos."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate(
    output: Path,
    n_rows: int = 12_000,
    n_users: int = 800,
    n_items: int = 1_200,
    seed: int = 42,
) -> int:
    """Write synthetic interactions to ``output``.

    Args:
        output: Destination CSV path.
        n_rows: Number of interaction rows.
        n_users: Distinct user pool size.
        n_items: Distinct item pool size.
        seed: RNG seed for reproducibility.

    Returns:
        int: Number of rows written (excluding header).
    """
    random.seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    events = ["view", "click", "cart", "purchase"]
    weights = [0.55, 0.25, 0.12, 0.08]
    start = datetime(2025, 1, 1)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "item_id", "event_type", "rating", "timestamp"],
        )
        writer.writeheader()
        for _ in range(n_rows):
            event = random.choices(events, weights=weights, k=1)[0]
            rating = ""
            if event == "purchase":
                rating = f"{random.uniform(3.0, 5.0):.2f}"
            elif event == "cart":
                rating = f"{random.uniform(2.0, 4.5):.2f}"
            ts = start + timedelta(minutes=random.randint(0, 60 * 24 * 180))
            writer.writerow(
                {
                    "user_id": f"U{random.randint(1, n_users):04d}",
                    "item_id": f"I{random.randint(1, n_items):04d}",
                    "event_type": event,
                    "rating": rating,
                    "timestamp": ts.isoformat(timespec="seconds"),
                }
            )
    return n_rows


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/interactions.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--n-rows", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n = generate(args.output, n_rows=args.n_rows, seed=args.seed)
    print(f"wrote {n} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
