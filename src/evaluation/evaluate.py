"""Model evaluation module."""

from typing import Any

import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import ModelEvaluationError


class ModelEvaluator(PipelineStage):
    """Evaluate trained model against promotion thresholds.

    This class evaluates a trained model and determines if it should
    be promoted to the Model Registry based on predefined thresholds.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the ModelEvaluator.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)
        self.thresholds = self.config.evaluation.thresholds
        self.last_metrics: dict[str, float] = {}

    def run(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> bool:
        """Evaluate model and determine if it should be promoted.

        Args:
            model: Trained model to evaluate.
            X_test: Test features.
            y_test: Test target.

        Returns:
            True if model meets all promotion thresholds, False otherwise.

        Raises:
            ModelEvaluationError: If evaluation fails.
        """
        try:
            self.logger.info(
                "Starting model evaluation",
                test_rows=len(X_test),
            )

            metrics = self._calculate_metrics(model, X_test, y_test)
            self.last_metrics = metrics

            self.logger.info("Calculated metrics", metrics=metrics)

            passed = self._check_thresholds(metrics)

            self.logger.info(
                "Model evaluation completed",
                passed=passed,
                metrics=metrics,
                thresholds=self.thresholds,
            )

            return passed

        except ModelEvaluationError:
            raise
        except Exception as e:
            raise ModelEvaluationError(
                f"Model evaluation failed: {e!s}",
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

    def _check_thresholds(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet promotion thresholds.

        Args:
            metrics: Calculated metrics.

        Returns:
            True if all metrics meet thresholds, False otherwise.
        """
        for metric_name, threshold in self.thresholds.items():
            if metric_name not in metrics:
                self.logger.warning(
                    "Metric not found",
                    metric=metric_name,
                )
                return False

            if metrics[metric_name] < threshold:
                self.logger.warning(
                    "Metric below threshold",
                    metric=metric_name,
                    value=metrics[metric_name],
                    threshold=threshold,
                )
                return False

        return True
