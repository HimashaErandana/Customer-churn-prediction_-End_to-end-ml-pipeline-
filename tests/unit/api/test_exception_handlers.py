"""Unit tests for exception_handlers module."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.exception_handlers import register_exception_handlers
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


def _create_test_app() -> FastAPI:
    """Create a FastAPI app with exception handlers and test routes."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/error/ml-pipeline")
    async def raise_ml_pipeline_error():
        raise MLPipelineError(message="General pipeline error", details={"key": "value"})

    @app.get("/error/data-ingestion")
    async def raise_data_ingestion_error():
        raise DataIngestionError(message="Ingestion failed", details={"file": "data.csv"})

    @app.get("/error/data-validation")
    async def raise_data_validation_error():
        raise DataValidationError(message="Validation failed", details={"column": "age"})

    @app.get("/error/data-processing")
    async def raise_data_processing_error():
        raise DataProcessingError(message="Processing failed", details={"step": "clean"})

    @app.get("/error/feature-engineering")
    async def raise_feature_engineering_error():
        raise FeatureEngineeringError(message="Feature failed", details={"feature": "x"})

    @app.get("/error/model-training")
    async def raise_model_training_error():
        raise ModelTrainingError(message="Training failed", details={"model": "xgb"})

    @app.get("/error/model-evaluation")
    async def raise_model_evaluation_error():
        raise ModelEvaluationError(message="Evaluation failed", details={"metric": "auc"})

    @app.get("/error/model-load")
    async def raise_model_load_error():
        raise ModelLoadError(message="Load failed", details={"path": "model.pkl"})

    @app.get("/error/prediction")
    async def raise_prediction_error():
        raise PredictionError(message="Prediction failed", details={"record": "1"})

    return app


class TestExceptionHandlers:
    """Tests for exception handlers returning correct status codes."""

    def test_ml_pipeline_error_handler(self) -> None:
        """Test MLPipelineError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/ml-pipeline")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "ML Pipeline Error"
        assert body["message"] == "General pipeline error"
        assert body["details"] == {"key": "value"}

    def test_data_ingestion_error_handler(self) -> None:
        """Test DataIngestionError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/data-ingestion")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Data Ingestion Error"
        assert body["message"] == "Ingestion failed"

    def test_data_validation_error_handler(self) -> None:
        """Test DataValidationError returns 422."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/data-validation")
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "Data Validation Error"
        assert body["message"] == "Validation failed"

    def test_data_processing_error_handler(self) -> None:
        """Test DataProcessingError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/data-processing")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Data Processing Error"

    def test_feature_engineering_error_handler(self) -> None:
        """Test FeatureEngineeringError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/feature-engineering")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Feature Engineering Error"

    def test_model_training_error_handler(self) -> None:
        """Test ModelTrainingError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/model-training")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Model Training Error"

    def test_model_evaluation_error_handler(self) -> None:
        """Test ModelEvaluationError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/model-evaluation")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Model Evaluation Error"

    def test_model_load_error_handler(self) -> None:
        """Test ModelLoadError returns 503."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/model-load")
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "Model Load Error"
        assert body["message"] == "Load failed"

    def test_prediction_error_handler(self) -> None:
        """Test PredictionError returns 500."""
        app = _create_test_app()
        client = TestClient(app)
        response = client.get("/error/prediction")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "Prediction Error"
        assert body["message"] == "Prediction failed"
