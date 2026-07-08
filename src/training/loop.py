"""Training loop with early stopping for the MLP recommender."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.mlp import MLPRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PairDataset(Dataset):
    """Positive pairs plus on-the-fly negative samples."""

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        n_items: int,
        neg_per_pos: int = 1,
        seed: int = 42,
    ) -> None:
        self.users = users.astype(np.int64)
        self.items = items.astype(np.int64)
        self.n_items = n_items
        self.neg_per_pos = neg_per_pos
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[int, int, float]:
        return int(self.users[index]), int(self.items[index]), 1.0

    def collate(self, batch: list[tuple[int, int, float]]) -> dict[str, torch.Tensor]:
        """Build tensors with negative samples for BCE."""
        pos_u = [b[0] for b in batch]
        pos_i = [b[1] for b in batch]
        users = pos_u.copy()
        items = pos_i.copy()
        labels = [1.0] * len(batch)
        for u, i in zip(pos_u, pos_i, strict=True):
            for _ in range(self.neg_per_pos):
                neg = int(self.rng.integers(0, self.n_items))
                while neg == i:
                    neg = int(self.rng.integers(0, self.n_items))
                users.append(u)
                items.append(neg)
                labels.append(0.0)
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "items": torch.tensor(items, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


@dataclass
class TrainResult:
    """Summary of a training run."""

    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float]]


def _split_val(
    frame: pd.DataFrame,
    ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = frame.sample(frac=1.0, random_state=seed)
    n_val = max(1, int(len(shuffled) * ratio))
    return shuffled.iloc[n_val:], shuffled.iloc[:n_val]


def _epoch_loss(
    model: MLPRecommender,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    assert model.net is not None
    training = optimizer is not None
    model.net.train(training)
    total = 0.0
    n = 0
    for batch in loader:
        users = batch["users"]
        items = batch["items"]
        labels = batch["labels"]
        logits = model.net(users, items)
        loss = criterion(logits, labels)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(labels)
        n += len(labels)
    return total / max(n, 1)


def train_mlp(
    train_df: pd.DataFrame,
    *,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    batch_size: int = 256,
    max_epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.1,
    neg_per_pos: int = 1,
    seed: int = 42,
) -> tuple[MLPRecommender, TrainResult]:
    """Train :class:`MLPRecommender` with early stopping on val BCE.

    Args:
        train_df: Feature table with ``user_idx`` / ``item_idx``.
        embedding_dim: Embedding size.
        hidden_dim: MLP hidden size.
        batch_size: Mini-batch size.
        max_epochs: Maximum epochs.
        patience: Early-stopping patience.
        learning_rate: Adam learning rate.
        val_ratio: Hold-out fraction from train for validation.
        neg_per_pos: Negatives sampled per positive.
        seed: RNG seed.

    Returns:
        tuple: Fitted model and training summary.
    """
    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["item_idx"].max()) + 1
    fit_df, val_df = _split_val(train_df, val_ratio, seed)

    model = MLPRecommender(
        name="mlp",
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_users=n_users,
        n_items=n_items,
    )
    model.build(n_users, n_items)
    assert model.net is not None

    train_ds = PairDataset(
        fit_df["user_idx"].to_numpy(),
        fit_df["item_idx"].to_numpy(),
        n_items=n_items,
        neg_per_pos=neg_per_pos,
        seed=seed,
    )
    val_ds = PairDataset(
        val_df["user_idx"].to_numpy(),
        val_df["item_idx"].to_numpy(),
        n_items=n_items,
        neg_per_pos=neg_per_pos,
        seed=seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_ds.collate,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=learning_rate)

    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _epoch_loss(model, train_loader, criterion, optimizer)
        val_loss = _epoch_loss(model, val_loader, criterion, None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f",
            epoch,
            train_loss,
            val_loss,
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.net.state_dict().items()
            }
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "early stopping at epoch=%s best_epoch=%s",
                    epoch,
                    best_epoch,
                )
                break

    if best_state is not None:
        model.net.load_state_dict(best_state)
    model.is_fitted = True
    return model, TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        history=history,
    )
