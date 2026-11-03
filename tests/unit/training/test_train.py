"""Unit tests for model training module."""

import pandas as pd
from xgboost import XGBClassifier

from src.training.train import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def _create_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create test data for training."""
        import numpy as np

        np.random.seed(42)
        n_samples = 200

        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            }
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

        return X, y

    def test_train_success(self, monkeypatch: object) -> None:
        """Test successful model training."""
        X, y = self._create_test_data()

        class FakeRun:
            def __enter__(self) -> "FakeRun":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def mock_value(*args: object, **kwargs: object) -> None:
            pass

        def mock_start_run(*args: object, **kwargs: object) -> FakeRun:
            return FakeRun()

        monkeypatch.setattr("mlflow.set_tracking_uri", mock_value)  # type: ignore
        monkeypatch.setattr("mlflow.set_experiment", mock_value)  # type: ignore
        monkeypatch.setattr("mlflow.start_run", mock_start_run)  # type: ignore
        monkeypatch.setattr("mlflow.log_params", mock_value)  # type: ignore
        monkeypatch.setattr("mlflow.log_metrics", mock_value)  # type: ignore
        monkeypatch.setattr("mlflow.xgboost.log_model", mock_value)  # type: ignore

        trainer = ModelTrainer()
        params = {"n_estimators": 10, "max_depth": 3, "random_state": 42}

        model, metrics = trainer.run(X, y, params)

        assert isinstance(model, XGBClassifier)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
