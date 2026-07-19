"""Smoke tests for the serving API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ARTIFACTS = Path("artifacts/serving")
MODEL = ARTIFACTS / "recommender.pt"
MAPS = ARTIFACTS / "id_maps.json"

pytestmark = pytest.mark.skipif(
    not MODEL.exists() or not MAPS.exists(),
    reason="serving artifacts missing",
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MODEL_PATH", str(MODEL))
    monkeypatch.setenv("ID_MAPS_PATH", str(MAPS))
    from src.api.service import load_bundle

    load_bundle.cache_clear()
    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client
    load_bundle.cache_clear()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommend_known_user(client: TestClient) -> None:
    response = client.get("/recommend", params={"user_id": "U0001", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "U0001"
    assert len(body["items"]) == 5
    assert "item_id" in body["items"][0]
    assert "score" in body["items"][0]


def test_recommend_unknown_user(client: TestClient) -> None:
    response = client.get("/recommend", params={"user_id": "U999999"})
    assert response.status_code == 404
