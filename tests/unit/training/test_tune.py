"""Unit tests for hyperparameter tuning module."""

import pandas as pd

from src.training.tune import HyperparameterTuner


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def _create_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Create test data for tuning."""
        import numpy as np

        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
            }
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

        return X, y

    def test_tune_success(self) -> None:
        """Test successful hyperparameter tuning."""
        X, y = self._create_test_data()

        tuner = HyperparameterTuner()
        best_params = tuner.run(X, y)

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params
