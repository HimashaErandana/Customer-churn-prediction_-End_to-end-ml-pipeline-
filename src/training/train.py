"""Model training module with MLflow logging."""

from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from src.config.pipeline_config import PipelineConfig
from src.config.settings import settings
from src.pipeline.base import PipelineStage
from src.utils.exceptions import ModelTrainingError


class ModelTrainer(PipelineStage):
    """Train XGBoost model and log to MLflow.

    This class handles model training, including data splitting,
    model fitting, and logging metrics/parameters to MLflow.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the ModelTrainer.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any],
        experiment_name: str | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Train XGBoost model and return model with metrics.

        Trains on all of the data it is given. An internal validation subset
        is carved from it for early-stopping monitoring and for the metrics
        returned here - the promotion gate must always be run by the caller
        on a separate, untouched holdout.

        Args:
            X: Feature matrix.
            y: Target vector.
            params: Model hyperparameters.
            experiment_name: Optional MLflow experiment name override.

        Returns:
            Tuple of (model, metrics) where model is the trained XGBoost model
            and metrics is a dictionary of evaluation metrics computed on the
            internal validation subset.

        Raises:
            ModelTrainingError: If training fails.
        """
        try:
            self.logger.info(
                "Starting model training",
                features_shape=X.shape,
                target_shape=y.shape,
                params=params,
                experiment=experiment_name,
            )

            X_fit, X_val, y_fit, y_val = train_test_split(
                X,
                y,
                test_size=self.config.data.validation_size,
                random_state=self.config.data.random_state,
                stratify=y,
            )

            self.logger.info(
                "Split data",
                fit_rows=len(X_fit),
                val_rows=len(X_val),
            )

            experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                mlflow.log_params(params)

                model = XGBClassifier(**params)
                model.fit(
                    X_fit,
                    y_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                metrics = self._calculate_metrics(model, X_val, y_val)

                mlflow.log_metrics(metrics)

                mlflow.xgboost.log_model(model, name="model")

                self.logger.info(
                    "Model training completed",
                    metrics=metrics,
                    experiment=experiment_name,
                )

                return model, metrics

        except ModelTrainingError:
            raise
        except Exception as e:
            raise ModelTrainingError(
                f"Model training failed: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _calculate_metrics(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of evaluation metrics.
        """
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        return metrics
