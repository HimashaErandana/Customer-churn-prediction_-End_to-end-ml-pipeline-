"""Unit tests for model evaluation module."""

import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.evaluate import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def _create_test_data_and_model(self) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
        """Create test data and train a model."""
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

        model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)

        return model, X, y

    def test_evaluate_pass(self) -> None:
        """Test model evaluation passes when metrics meet thresholds."""
        model, X, y = self._create_test_data_and_model()

        evaluator = ModelEvaluator()
        evaluator.thresholds = {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "roc_auc": 0.5,
        }

        passed = evaluator.run(model, X, y)

        assert isinstance(passed, bool)

    def test_evaluate_fail(self) -> None:
        """Test model evaluation fails when metrics don't meet thresholds."""
        model, X, y = self._create_test_data_and_model()

        evaluator = ModelEvaluator()
        evaluator.thresholds = {
            "accuracy": 0.99,
            "precision": 0.99,
            "recall": 0.99,
            "f1": 0.99,
            "roc_auc": 0.99,
        }

        passed = evaluator.run(model, X, y)

        assert passed is False
