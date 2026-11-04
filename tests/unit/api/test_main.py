"""Unit tests for API module."""

from fastapi.testclient import TestClient

import numpy as np


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self) -> None:
        """Test health check endpoint."""
        from api.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_validation_error(self) -> None:
        """Test prediction endpoint with invalid data."""
        from api.main import app

        client = TestClient(app)
        response = client.post(
            "/predict",
            json={
                "gender": "Invalid",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            },
        )

        assert response.status_code == 422

    def test_predict_success(self) -> None:
        """Test successful single prediction."""
        from api.main import app
        from unittest.mock import patch, MagicMock

        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model.return_value = mock_model
            mock_loader.get_label_encoders.return_value = {}
            mock_loader.get_model_version.return_value = "xgboost_v3"

            client = TestClient(app)
            response = client.post(
                "/predict",
                json={
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "Yes",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == 1
        assert body["probability"] == 0.8
        assert body["model_version"] == "xgboost_v3"

    def test_predict_model_load_error(self) -> None:
        """Test prediction endpoint when model fails to load."""
        from api.main import app
        from unittest.mock import patch
        from src.utils.exceptions import ModelLoadError

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model.side_effect = ModelLoadError(
                message="Model not found", details={"path": "model.pkl"}
            )

            client = TestClient(app)
            response = client.post(
                "/predict",
                json={
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "No",
                    "DeviceProtection": "Yes",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                },
            )

        assert response.status_code == 503


class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_empty(self) -> None:
        """Test batch prediction with empty records."""
        from api.main import app

        client = TestClient(app)
        response = client.post(
            "/predict/batch",
            json={"records": []},
        )

        assert response.status_code == 200
        assert response.json() == {"predictions": []}

    def test_batch_predict_success(self) -> None:
        """Test successful batch prediction."""
        from api.main import app
        from unittest.mock import patch, MagicMock

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model.return_value = mock_model
            mock_loader.get_label_encoders.return_value = {}
            mock_loader.get_model_version.return_value = "xgboost_v3"

            client = TestClient(app)
            response = client.post(
                "/predict/batch",
                json={
                    "records": [
                        {
                            "gender": "Female",
                            "SeniorCitizen": 0,
                            "Partner": "Yes",
                            "Dependents": "No",
                            "tenure": 12,
                            "PhoneService": "Yes",
                            "MultipleLines": "No",
                            "InternetService": "DSL",
                            "OnlineSecurity": "Yes",
                            "OnlineBackup": "No",
                            "DeviceProtection": "Yes",
                            "TechSupport": "No",
                            "StreamingTV": "No",
                            "StreamingMovies": "No",
                            "Contract": "Month-to-month",
                            "PaperlessBilling": "Yes",
                            "PaymentMethod": "Electronic check",
                            "MonthlyCharges": 29.85,
                            "TotalCharges": 29.85,
                        },
                        {
                            "gender": "Male",
                            "SeniorCitizen": 1,
                            "Partner": "No",
                            "Dependents": "Yes",
                            "tenure": 24,
                            "PhoneService": "No",
                            "MultipleLines": "No phone service",
                            "InternetService": "Fiber optic",
                            "OnlineSecurity": "No",
                            "OnlineBackup": "Yes",
                            "DeviceProtection": "No",
                            "TechSupport": "Yes",
                            "StreamingTV": "Yes",
                            "StreamingMovies": "No",
                            "Contract": "One year",
                            "PaperlessBilling": "No",
                            "PaymentMethod": "Credit card (automatic)",
                            "MonthlyCharges": 59.90,
                            "TotalCharges": 1437.50,
                        },
                    ],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2
        assert body["predictions"][0]["prediction"] == 1
        assert body["predictions"][0]["probability"] == 0.8
        assert body["predictions"][1]["prediction"] == 0
        assert body["predictions"][1]["probability"] == 0.3

    def test_batch_predict_model_load_error(self) -> None:
        """Test batch prediction endpoint when model fails to load."""
        from api.main import app
        from unittest.mock import patch
        from src.utils.exceptions import ModelLoadError

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model.side_effect = ModelLoadError(
                message="Model not found", details={"path": "model.pkl"}
            )

            client = TestClient(app)
            response = client.post(
                "/predict/batch",
                json={
                    "records": [
                        {
                            "gender": "Female",
                            "SeniorCitizen": 0,
                            "Partner": "Yes",
                            "Dependents": "No",
                            "tenure": 12,
                            "PhoneService": "Yes",
                            "MultipleLines": "No",
                            "InternetService": "DSL",
                            "OnlineSecurity": "Yes",
                            "OnlineBackup": "No",
                            "DeviceProtection": "Yes",
                            "TechSupport": "No",
                            "StreamingTV": "No",
                            "StreamingMovies": "No",
                            "Contract": "Month-to-month",
                            "PaperlessBilling": "Yes",
                            "PaymentMethod": "Electronic check",
                            "MonthlyCharges": 29.85,
                            "TotalCharges": 29.85,
                        },
                    ],
                },
            )

        assert response.status_code == 503


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_success(self) -> None:
        """Test successful model info retrieval."""
        from api.main import app
        from unittest.mock import patch

        mock_info = {
            "version": "xgboost_v3",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"roc_auc": 0.85, "accuracy": 0.8},
        }

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model_info.return_value = mock_info

            client = TestClient(app)
            response = client.get("/model/info")

        assert response.status_code == 200
        body = response.json()
        assert body["version"] == "xgboost_v3"
        assert body["trained_at"] == "2026-01-01T00:00:00+00:00"
        assert body["metrics"]["roc_auc"] == 0.85

    def test_model_info_model_load_error(self) -> None:
        """Test model info endpoint when model fails to load."""
        from api.main import app
        from unittest.mock import patch
        from src.utils.exceptions import ModelLoadError

        with patch("api.main.model_loader") as mock_loader:
            mock_loader.get_model_info.side_effect = ModelLoadError(
                message="Model not loaded", details={"model_version": "unknown"}
            )

            client = TestClient(app)
            response = client.get("/model/info")

        assert response.status_code == 503
