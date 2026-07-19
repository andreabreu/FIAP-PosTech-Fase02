"""FastAPI app — health + recommendations."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.service import load_bundle, recommend_for_user


class RecommendResponse(BaseModel):
    """Top-k recommendation payload."""

    user_id: str
    k: int = Field(ge=1, le=100)
    items: list[dict[str, Any]]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm model on startup so first request is fast."""
    model_path = os.getenv("MODEL_PATH", "artifacts/serving/recommender.pt")
    maps_path = os.getenv("ID_MAPS_PATH", "artifacts/serving/id_maps.json")
    load_bundle(model_path, maps_path)
    yield


app = FastAPI(
    title="FIAP Fase 02 Recommender",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for Container Apps."""
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: str = Query(..., examples=["U0001"]),
    k: int = Query(10, ge=1, le=100),
) -> RecommendResponse:
    """Top-k para um user_id."""
    try:
        items = recommend_for_user(user_id, k=k)
    except KeyError as exc:
        detail = f"unknown user_id: {user_id}"
        raise HTTPException(status_code=404, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendResponse(user_id=user_id, k=k, items=items)
