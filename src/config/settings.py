"""Runtime settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for local and container runs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = Field(default="fiap-postech-fase02")
    random_seed: int = Field(default=42)
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    model_name: str = Field(default="mlp")
    embedding_dim: int = Field(default=32)
    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000")


def get_settings() -> Settings:
    """Build settings from env/.env.

    Returns:
        Settings: Parsed settings instance.
    """
    return Settings()
