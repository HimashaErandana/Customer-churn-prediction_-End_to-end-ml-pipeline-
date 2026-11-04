"""Custom exceptions for the ML pipeline."""

from typing import Any


class MLPipelineError(Exception):
    """Base exception for all ML pipeline errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataIngestionError(MLPipelineError):
    """Raised when data ingestion fails."""

    pass


class DataValidationError(MLPipelineError):
    """Raised when data validation fails."""

    pass


class DataProcessingError(MLPipelineError):
    """Raised when data processing fails."""

    pass


class FeatureEngineeringError(MLPipelineError):
    """Raised when feature engineering fails."""

    pass


class ModelTrainingError(MLPipelineError):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(MLPipelineError):
    """Raised when model evaluation fails."""

    pass


class ModelLoadError(MLPipelineError):
    """Raised when model loading fails."""

    pass


class PredictionError(MLPipelineError):
    """Raised when prediction fails."""

    pass


class ConfigurationError(MLPipelineError):
    """Raised when configuration is invalid."""

    pass
