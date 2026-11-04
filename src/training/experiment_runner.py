"""Experiment runner for hyperparameter experimentation."""

from typing import Any

import mlflow
import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.training.train import ModelTrainer
from src.utils.exceptions import ModelTrainingError


class ExperimentRunner(PipelineStage):
    """Run multiple hyperparameter experiments and log each to MLflow.

    Iterates through the experiments defined in config.yaml, trains a model
    for each under its own MLflow experiment, and returns the results.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the ExperimentRunner.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
        """Run all configured experiments.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            List of experiment results, each containing the experiment name,
            description, hyperparameters, metrics, model, and MLflow run id.

        Raises:
            ModelTrainingError: If any experiment fails.
        """
        try:
            results: list[dict[str, Any]] = []

            for experiment in self.config.model.experiments:
                self.logger.info(
                    "Starting experiment",
                    experiment=experiment.name,
                    hyperparameters=experiment.hyperparameters,
                )

                mlflow.set_experiment(experiment.name)

                trainer = ModelTrainer(self.config)
                model, metrics = trainer.run(
                    X, y, experiment.hyperparameters, experiment_name=experiment.name
                )

                last_run = mlflow.last_active_run()
                if last_run is None or last_run.info is None:
                    raise ModelTrainingError(
                        "No active MLflow run found after training",
                        details={"experiment": experiment.name},
                    )
                run_id = last_run.info.run_id

                results.append(
                    {
                        "experiment_name": experiment.name,
                        "description": experiment.description,
                        "hyperparameters": experiment.hyperparameters,
                        "metrics": metrics,
                        "model": model,
                        "run_id": run_id,
                    }
                )

                self.logger.info(
                    "Experiment completed",
                    experiment=experiment.name,
                    metrics=metrics,
                    run_id=run_id,
                )

            return results

        except ModelTrainingError:
            raise
        except Exception as e:
            raise ModelTrainingError(
                f"Experiment runner failed: {e!s}",
                details={"error": str(e)},
            ) from e
