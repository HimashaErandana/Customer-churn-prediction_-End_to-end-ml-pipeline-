"""Unit tests for API module."""

from fastapi.testclient import TestClient


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
