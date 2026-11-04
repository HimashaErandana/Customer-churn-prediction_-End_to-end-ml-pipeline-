"""Exception handlers for the FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.utils.exceptions import (
    DataIngestionError,
    DataProcessingError,
    DataValidationError,
    FeatureEngineeringError,
    MLPipelineError,
    ModelEvaluationError,
    ModelLoadError,
    ModelTrainingError,
    PredictionError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Register exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance.
    """

    @app.exception_handler(MLPipelineError)
    async def ml_pipeline_error_handler(
        request: Request, exc: MLPipelineError
    ) -> JSONResponse:
        """Handle ML pipeline errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "ML Pipeline Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(DataIngestionError)
    async def data_ingestion_error_handler(
        request: Request, exc: DataIngestionError
    ) -> JSONResponse:
        """Handle data ingestion errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Data Ingestion Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(DataValidationError)
    async def data_validation_error_handler(
        request: Request, exc: DataValidationError
    ) -> JSONResponse:
        """Handle data validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Data Validation Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(DataProcessingError)
    async def data_processing_error_handler(
        request: Request, exc: DataProcessingError
    ) -> JSONResponse:
        """Handle data processing errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Data Processing Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(FeatureEngineeringError)
    async def feature_engineering_error_handler(
        request: Request, exc: FeatureEngineeringError
    ) -> JSONResponse:
        """Handle feature engineering errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Feature Engineering Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ModelTrainingError)
    async def model_training_error_handler(
        request: Request, exc: ModelTrainingError
    ) -> JSONResponse:
        """Handle model training errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Training Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ModelEvaluationError)
    async def model_evaluation_error_handler(
        request: Request, exc: ModelEvaluationError
    ) -> JSONResponse:
        """Handle model evaluation errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model Evaluation Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ModelLoadError)
    async def model_load_error_handler(
        request: Request, exc: ModelLoadError
    ) -> JSONResponse:
        """Handle model load errors."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model Load Error",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request, exc: PredictionError
    ) -> JSONResponse:
        """Handle prediction errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction Error",
                "message": exc.message,
                "details": exc.details,
            },
        )
