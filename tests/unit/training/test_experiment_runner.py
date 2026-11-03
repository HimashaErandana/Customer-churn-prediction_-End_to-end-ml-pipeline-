"""Unit tests for experiment runner module."""

import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.training.experiment_runner import ExperimentRunner


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

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

    def _make_config(self) -> PipelineConfig:
        """Build a minimal PipelineConfig with two experiments."""
        return PipelineConfig(
            project_name="churn-prediction",
            model_type="xgboost",
            data={
                "target_column": "Churn",
                "positive_class": "Yes",
                "negative_class": "No",
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42,
            },
            features={
                "numerical_columns": ["feature1", "feature2", "feature3"],
                "categorical_columns": [],
                "drop_columns": [],
            },
            model={
                "name": "xgboost",
                "registered_model_name": "CustomerChurnXGBoost",
                "xgboost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "gamma": 0,
                    "reg_alpha": 0,
                    "reg_lambda": 1,
                    "random_state": 42,
                },
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                },
                "experiments": [
                    {
                        "name": "xgboost_v1_baseline",
                        "description": "baseline",
                        "hyperparameters": {
                            "n_estimators": 10,
                            "max_depth": 3,
                            "learning_rate": 0.1,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "min_child_weight": 1,
                            "gamma": 0,
                            "reg_alpha": 0,
                            "reg_lambda": 1,
                            "random_state": 42,
                        },
                    },
                    {
                        "name": "xgboost_v2",
                        "description": "deeper",
                        "hyperparameters": {
                            "n_estimators": 20,
                            "max_depth": 5,
                            "learning_rate": 0.05,
                            "subsample": 0.8,
                            "colsample_bytree": 0.9,
                            "min_child_weight": 3,
                            "gamma": 0.1,
                            "reg_alpha": 0.01,
                            "reg_lambda": 2,
                            "random_state": 42,
                        },
                    },
                ],
            },
            training={
                "cv_folds": 5,
                "scoring": "roc_auc",
                "early_stopping_rounds": 10,
                "eval_metric": "logloss",
            },
            evaluation={
                "accuracy_threshold": 0.7,
                "precision_threshold": 0.7,
                "recall_threshold": 0.7,
                "f1_threshold": 0.7,
                "roc_auc_threshold": 0.75,
            },
            promotion={"threshold": 0.8, "metric": "roc_auc"},
            logging={"level": "INFO", "format": "%(message)s"},
        )

    def test_run_all_experiments(self, monkeypatch: object) -> None:
        """Test that all experiments run and return results."""
        X, y = self._create_test_data()
        config = self._make_config()

        class FakeRun:
            def __enter__(self) -> "FakeRun":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeMLflowRun:
            class Info:
                run_id = "run_123"

            info = Info()

        def mock_set_tracking_uri(*args: object, **kwargs: object) -> None:
            pass

        def mock_set_experiment(*args: object, **kwargs: object) -> None:
            pass

        def mock_start_run(*args: object, **kwargs: object) -> FakeRun:
            return FakeRun()

        def mock_log_params(*args: object, **kwargs: object) -> None:
            pass

        def mock_log_metrics(*args: object, **kwargs: object) -> None:
            pass

        def mock_log_model(*args: object, **kwargs: object) -> None:
            pass

        def mock_last_active_run() -> FakeMLflowRun:
            return FakeMLflowRun()

        monkeypatch.setattr("mlflow.set_tracking_uri", mock_set_tracking_uri)
        monkeypatch.setattr("mlflow.set_experiment", mock_set_experiment)
        monkeypatch.setattr("mlflow.start_run", mock_start_run)
        monkeypatch.setattr("mlflow.log_params", mock_log_params)
        monkeypatch.setattr("mlflow.log_metrics", mock_log_metrics)
        monkeypatch.setattr("mlflow.xgboost.log_model", mock_log_model)
        monkeypatch.setattr("mlflow.last_active_run", mock_last_active_run)

        runner = ExperimentRunner(config)
        results = runner.run(X, y)

        assert len(results) == 2
        names = {r["experiment_name"] for r in results}
        assert names == {"xgboost_v1_baseline", "xgboost_v2"}

        for result in results:
            assert "description" in result
            assert "hyperparameters" in result
            assert "metrics" in result
            assert "model" in result
            assert "run_id" in result
            assert result["run_id"] == "run_123"
            metrics = result["metrics"]
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                assert key in metrics
