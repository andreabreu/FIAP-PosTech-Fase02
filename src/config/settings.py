"""Runtime settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for local and container runs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    project_name: str = Field(default="fiap-postech-fase02")
    environment: str = Field(default="local")
    random_seed: int = Field(default=42, ge=0)
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    models_dir: str = Field(default="models")
    metrics_dir: str = Field(default="metrics")
    model_name: str = Field(default="mlp")
    embedding_dim: int = Field(default=32, ge=1)
    batch_size: int = Field(default=256, ge=1)
    max_epochs: int = Field(default=20, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000")
    mlflow_experiment_name: str = Field(default="recommender-fase02")
    log_level: str = Field(default="INFO")

    @field_validator(
        "raw_data_dir",
        "processed_data_dir",
        "models_dir",
        "metrics_dir",
    )
    @classmethod
    def normalize_path(cls, value: str) -> str:
        """Normalize configured directories to posix-like relative paths.

        Args:
            value: Raw directory path from env.

        Returns:
            str: Normalized path string.
        """
        return Path(value).as_posix()

    def ensure_directories(self) -> None:
        """Create configured directories when missing."""
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.metrics_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Build settings from env/.env.

    Returns:
        Settings: Parsed settings instance.
    """
    return Settings()
