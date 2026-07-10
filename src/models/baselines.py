"""Scikit-Learn style baselines for recommender comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.domain.interfaces import RecommenderModel


class PopularityRecommender(RecommenderModel):
    """Rank items by global interaction frequency."""

    def __init__(self, name: str = "popularity", **hparams: Any) -> None:
        self.name = name
        self.hparams = hparams
        self.item_scores: dict[int, float] = {}
        self.n_items = 0
        self.is_fitted = False

    def fit(self, interactions: Any) -> PopularityRecommender:
        """Fit popularity counts from training interactions."""
        frame = _as_frame(interactions)
        counts = frame.groupby("item_idx").size()
        self.n_items = int(frame["item_idx"].max()) + 1
        total = float(counts.sum()) or 1.0
        self.item_scores = {int(i): float(c) / total for i, c in counts.items()}
        self.is_fitted = True
        return self

    def predict(self, user_ids: Any, item_ids: Any) -> list[float]:
        """Return popularity scores (user ignored)."""
        _ = user_ids
        return [float(self.item_scores.get(int(i), 0.0)) for i in item_ids]

    def score_user_items(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Score candidates by popularity."""
        _ = user_idx
        return np.array(
            [self.item_scores.get(int(i), 0.0) for i in item_indices],
            dtype=np.float64,
        )


class SVDRecommender(RecommenderModel):
    """Matrix-factorization baseline via TruncatedSVD on user-item CSR."""

    def __init__(
        self,
        name: str = "svd",
        n_components: int = 32,
        **hparams: Any,
    ) -> None:
        self.name = name
        self.n_components = n_components
        self.hparams = hparams
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.n_users = 0
        self.n_items = 0
        self.is_fitted = False

    def fit(self, interactions: Any) -> SVDRecommender:
        """Factorize the implicit user-item matrix."""
        frame = _as_frame(interactions)
        self.n_users = int(frame["user_idx"].max()) + 1
        self.n_items = int(frame["item_idx"].max()) + 1
        rows = frame["user_idx"].to_numpy()
        cols = frame["item_idx"].to_numpy()
        if "score" in frame:
            data = frame["score"].to_numpy(dtype=np.float64)
        else:
            data = np.ones(len(frame))
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        n_comp = min(self.n_components, min(self.n_users, self.n_items) - 1)
        n_comp = max(n_comp, 2)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        self.is_fitted = True
        return self

    def predict(self, user_ids: Any, item_ids: Any) -> list[float]:
        """Dot-product scores for pairs."""
        if self.user_factors is None or self.item_factors is None:
            return [0.0] * len(list(user_ids))
        scores = []
        for u, i in zip(user_ids, item_ids, strict=False):
            scores.append(float(self.user_factors[int(u)] @ self.item_factors[int(i)]))
        return scores

    def score_user_items(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """Score many items for one user via reconstructed factors."""
        if self.user_factors is None or self.item_factors is None:
            return np.zeros(len(item_indices), dtype=np.float64)
        user_vec = self.user_factors[int(user_idx)]
        return self.item_factors[item_indices] @ user_vec


def _as_frame(interactions: Any) -> pd.DataFrame:
    if isinstance(interactions, pd.DataFrame):
        return interactions
    return pd.DataFrame(list(interactions))
