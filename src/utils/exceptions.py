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


class DataValidationError(MLPipelineError):
    """Raised when data validation fails."""


class DataProcessingError(MLPipelineError):
    """Raised when data processing fails."""


class FeatureEngineeringError(MLPipelineError):
    """Raised when feature engineering fails."""


class ModelTrainingError(MLPipelineError):
    """Raised when model training fails."""


class ModelEvaluationError(MLPipelineError):
    """Raised when model evaluation fails."""


class ModelLoadError(MLPipelineError):
    """Raised when model loading fails."""


class PredictionError(MLPipelineError):
    """Raised when prediction fails."""


class ConfigurationError(MLPipelineError):
    """Raised when configuration is invalid."""
