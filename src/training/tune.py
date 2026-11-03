"""Hyperparameter tuning module."""

from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import ModelTrainingError


class HyperparameterTuner(PipelineStage):
    """Tune XGBoost hyperparameters using GridSearchCV.

    This class finds the best hyperparameters for the XGBoost model
    using cross-validation. It returns the best parameters only,
    not the trained model.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the HyperparameterTuner.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Find best hyperparameters using GridSearchCV.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Dictionary of best hyperparameters.

        Raises:
            ModelTrainingError: If tuning fails.
        """
        try:
            self.logger.info(
                "Starting hyperparameter tuning",
                features_shape=X.shape,
                target_shape=y.shape,
            )

            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            }

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            base_model = XGBClassifier(
                random_state=42,
                eval_metric="logloss",
            )

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0,
            )

            self.logger.info("Fitting GridSearchCV", param_grid=param_grid)

            grid_search.fit(X, y)

            best_params = grid_search.best_params_

            self.logger.info(
                "Hyperparameter tuning completed",
                best_params=best_params,
                best_score=grid_search.best_score_,
            )

            return best_params

        except Exception as e:
            raise ModelTrainingError(
                f"Hyperparameter tuning failed: {str(e)}",
                details={"error": str(e)},
            ) from e
