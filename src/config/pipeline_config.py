"""Typed pipeline configuration loaded from config.yaml."""

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class DataConfig(BaseModel):
    """Data splitting and target configuration."""

    target_column: str
    positive_class: str
    negative_class: str
    test_size: float
    validation_size: float
    random_state: int


class FeaturesConfig(BaseModel):
    """Feature column lists and columns to drop."""

    numerical_columns: list[str]
    categorical_columns: list[str]
    drop_columns: list[str]


class ExperimentConfig(BaseModel):
    """A single hyperparameter experiment configuration."""

    name: str
    description: str
    hyperparameters: dict[str, float | int]


class ModelConfig(BaseModel):
    """Hyperparameter sets per estimator."""

    name: str
    registered_model_name: str
    xgboost: dict[str, float | int]
    random_forest: dict[str, float | int]
    experiments: list[ExperimentConfig]


class TrainingConfig(BaseModel):
    """Training-time settings."""

    cv_folds: int
    scoring: str
    early_stopping_rounds: int
    eval_metric: str


class EvaluationConfig(BaseModel):
    """Per-metric promotion thresholds."""

    accuracy_threshold: float
    precision_threshold: float
    recall_threshold: float
    f1_threshold: float
    roc_auc_threshold: float

    @property
    def thresholds(self) -> dict[str, float]:
        """Map threshold fields to metric names used by ModelEvaluator.

        Returns:
            Dictionary of metric name to minimum acceptable value.
        """
        return {
            "accuracy": self.accuracy_threshold,
            "precision": self.precision_threshold,
            "recall": self.recall_threshold,
            "f1": self.f1_threshold,
            "roc_auc": self.roc_auc_threshold,
        }


class PromotionConfig(BaseModel):
    """Model promotion gate settings."""

    threshold: float
    metric: str


class LoggingConfig(BaseModel):
    """Logging level and format."""

    level: str
    format: str


class PipelineConfig(BaseModel):
    """Full pipeline configuration, loaded from config.yaml."""

    project_name: str
    model_type: str
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    promotion: PromotionConfig
    logging: LoggingConfig


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    """Load the pipeline configuration from a YAML file.

    Args:
        path: Path to the config file. Defaults to src/config/config.yaml.

    Returns:
        Validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the contents do not match the schema.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
