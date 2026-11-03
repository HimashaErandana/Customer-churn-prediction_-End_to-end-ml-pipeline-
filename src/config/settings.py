"""Application settings loaded from environment variables."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    ENVIRONMENT: str = "local"
    DEBUG: bool = False

    # Data paths
    DATA_RAW_PATH: str = "data/raw"
    DATA_PROCESSED_PATH: str = "data/processed"
    DATA_EXTERNAL_PATH: str = "data/external"

    # MLflow
    MLFLOW_TRACKING_URI: str = "./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "churn-prediction/xgboost"
    MLFLOW_ALLOW_FILE_STORE: bool = True
    MODEL_NAME: str = "churn-xgboost"
    MODEL_STAGE: str = "Production"
    MODEL_S3_BUCKET: str = ""

    # AWS
    AWS_REGION: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    ECR_REPOSITORY_URI: str = ""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_BATCH_SIZE: int = 500

    # Model promotion threshold
    MODEL_PROMOTION_THRESHOLD: float = 0.8

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()

# MLflow 3.x refuses to construct a filesystem-backend store unless this flag is
# set. pydantic-settings does not export .env into the process environment, so
# mirror it here — the single sanctioned os.environ touchpoint in the project.
# Every code path that opens an MLflow store imports this module first.
os.environ.setdefault(
    "MLFLOW_ALLOW_FILE_STORE", "true" if settings.MLFLOW_ALLOW_FILE_STORE else "false"
)
