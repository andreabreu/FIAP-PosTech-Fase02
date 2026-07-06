"""PyTorch MLP recommender with user/item embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.domain.interfaces import RecommenderModel


class MLPNet(nn.Module):
    """Two-tower style MLP over concatenated embeddings."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score user-item pairs.

        Args:
            users: User index tensor ``[B]``.
            items: Item index tensor ``[B]``.

        Returns:
            torch.Tensor: Raw logits ``[B]``.
        """
        x = torch.cat([self.user_emb(users), self.item_emb(items)], dim=-1)
        return self.mlp(x).squeeze(-1)


class MLPRecommender(RecommenderModel):
    """Trainable MLP recommender wrapping :class:`MLPNet`."""

    def __init__(
        self,
        name: str = "mlp",
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        n_users: int = 0,
        n_items: int = 0,
        **hparams: Any,
    ) -> None:
        self.name = name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_users = n_users
        self.n_items = n_items
        self.hparams = hparams
        self.net: MLPNet | None = None
        self.device = torch.device("cpu")
        self.is_fitted = False

    def build(self, n_users: int, n_items: int) -> None:
        """Allocate network weights for the known catalog size."""
        self.n_users = n_users
        self.n_items = n_items
        self.net = MLPNet(n_users, n_items, self.embedding_dim, self.hidden_dim)
        self.net.to(self.device)

    def fit(self, interactions: Any) -> MLPRecommender:
        """Mark as fitted when external trainer already trained ``self.net``.

        Args:
            interactions: Unused when weights come from :mod:`src.training.loop`.

        Returns:
            MLPRecommender: Self.
        """
        _ = interactions
        if self.net is None and self.n_users > 0 and self.n_items > 0:
            self.build(self.n_users, self.n_items)
        self.is_fitted = self.net is not None
        return self

    def predict(self, user_ids: Any, item_ids: Any) -> list[float]:
        """Score pairs with the fitted network.

        Args:
            user_ids: Iterable of user indices.
            item_ids: Iterable of item indices.

        Returns:
            list[float]: Predicted scores (sigmoid probabilities).
        """
        if self.net is None:
            return [0.0] * max(len(list(user_ids)), 0)
        users = torch.as_tensor(list(user_ids), dtype=torch.long, device=self.device)
        items = torch.as_tensor(list(item_ids), dtype=torch.long, device=self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(users, items)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        return [float(x) for x in probs]

    def score_user_items(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Score many items for one user.

        Args:
            user_idx: User index.
            item_indices: Candidate item indices.

        Returns:
            np.ndarray: Scores aligned with ``item_indices``.
        """
        if self.net is None or len(item_indices) == 0:
            return np.zeros(len(item_indices), dtype=np.float64)
        users = torch.full(
            (len(item_indices),),
            user_idx,
            dtype=torch.long,
            device=self.device,
        )
        items = torch.as_tensor(item_indices, dtype=torch.long, device=self.device)
        self.net.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.net(users, items)).cpu().numpy()
        return scores.astype(np.float64)

    def save(self, path: Path) -> None:
        """Persist network weights and metadata."""
        if self.net is None:
            raise RuntimeError("cannot save unbuilt model")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.net.state_dict(),
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "name": self.name,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> MLPRecommender:
        """Load a previously saved MLP recommender."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            name=payload.get("name", "mlp"),
            embedding_dim=int(payload["embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            n_users=int(payload["n_users"]),
            n_items=int(payload["n_items"]),
        )
        model.build(model.n_users, model.n_items)
        assert model.net is not None
        model.net.load_state_dict(payload["state_dict"])
        model.is_fitted = True
        return model
