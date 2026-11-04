"""FastAPI application for serving predictions."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal, cast

import pandas as pd
from fastapi import FastAPI

from api.exception_handlers import register_exception_handlers
from api.model_loader import model_loader
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config.settings import settings
from src.utils.exceptions import ModelLoadError, PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)
#main--yyyy

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model at startup."""
    logger.info("Starting API server")
    try:
        model_loader.load_production_model()
        logger.info("Model loaded successfully")
    except ModelLoadError as e:
        logger.error("Failed to load model", error=str(e))
    yield
    logger.info("Shutting down API server")


app = FastAPI(
    title="Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe endpoint.

    Returns:
        Health status.
    """
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Get model information.

    Returns:
        Model version and metadata.

    Raises:
        ModelLoadError: If model is not loaded.
    """
    try:
        info = model_loader.get_model_info()
        return ModelInfoResponse(
            version=info["version"],
            trained_at=info["trained_at"],
            metrics=info["metrics"],
        )
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"Failed to get model info: {e!s}") from e


def _encode_input_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """Apply label encoding to categorical columns.

    Args:
        input_data: Raw input DataFrame with string categoricals.

    Returns:
        DataFrame with encoded categorical columns.
    """
    encoders = model_loader.get_label_encoders()

    for col, encoder in encoders.items():
        if col in input_data.columns:
            input_data[col] = encoder.transform(input_data[col])

    return input_data


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a single prediction.

    Args:
        request: Prediction request data.

    Returns:
        Prediction response with class and probability.

    Raises:
        PredictionError: If prediction fails.
    """
    try:
        model = model_loader.get_model()

        input_data = pd.DataFrame([request.model_dump()])
        input_data = _encode_input_data(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return PredictionResponse(
            prediction=cast("Literal[0, 1]", int(prediction)),
            probability=float(probability),
            model_version=model_loader.get_model_version(),
        )
    except ModelLoadError:
        raise
    except Exception as e:
        raise PredictionError(f"Prediction failed: {e!s}") from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Make batch predictions.

    Args:
        request: Batch prediction request data.

    Returns:
        Batch prediction response.

    Raises:
        PredictionError: If prediction fails.
    """
    if not request.records:
        return BatchPredictionResponse(predictions=[])

    try:
        model = model_loader.get_model()

        input_data = pd.DataFrame([r.model_dump() for r in request.records])
        input_data = _encode_input_data(input_data)

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        results = []
        for pred, prob in zip(predictions, probabilities, strict=True):
            results.append(
                PredictionResponse(
                    prediction=cast("Literal[0, 1]", int(pred)),
                    probability=float(prob),
                    model_version=model_loader.get_model_version(),
                )
            )

        return BatchPredictionResponse(predictions=results)
    except ModelLoadError:
        raise
    except Exception as e:
        raise PredictionError(f"Batch prediction failed: {e!s}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
