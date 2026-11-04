"""Export the approved model as a deployment artifact.

This module bundles the selected model, preprocessing components (label
encoders), feature names, and metadata into a single deployment artifact
that can be copied into a Docker image for serving. MLflow is NOT used
at serving time.
"""

from pathlib import Path

import joblib  # type: ignore[import-untyped]

from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAINED_MODEL_PATH = Path("model/model.pkl")
DEPLOYMENT_DIR = Path("deployment/model")
DEPLOYMENT_ARTIFACT = DEPLOYMENT_DIR / "model.pkl"


def export_approved_model() -> None:
    """Export the evaluated model to the deployment directory.

    Reads the trained model artifact from ``model/model.pkl`` (produced by
    ``run_pipeline.py``), validates that it passed evaluation, and writes
    a clean deployment artifact to ``deployment/model/model.pkl``.

    Raises:
        ModelTrainingError: If the model did not pass evaluation or the
            artifact cannot be read.
    """
    try:
        if not TRAINED_MODEL_PATH.exists():
            raise ModelTrainingError(
                f"Trained model artifact not found: {TRAINED_MODEL_PATH}",
                details={"path": str(TRAINED_MODEL_PATH)},
            )

        model_data = joblib.load(TRAINED_MODEL_PATH)

        if not model_data.get("passed_evaluation", False):
            raise ModelTrainingError(
                "Model did not pass evaluation gate. Refusing to export.",
                details={"passed_evaluation": model_data.get("passed_evaluation")},
            )

        deployment_artifact = {
            "model": model_data["model"],
            "label_encoders": model_data.get("label_encoders", {}),
            "feature_names": model_data.get("feature_names", []),
            "metadata": {
                "best_experiment": model_data.get("best_experiment"),
                "train_metrics": model_data.get("train_metrics", {}),
                "gate_metrics": model_data.get("metrics", {}),
            },
        }

        DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(deployment_artifact, DEPLOYMENT_ARTIFACT)

        logger.info(
            "Model exported for deployment",
            path=str(DEPLOYMENT_ARTIFACT),
            experiment=model_data.get("best_experiment"),
            passed_evaluation=model_data.get("passed_evaluation"),
        )

    except ModelTrainingError:
        raise
    except Exception as e:
        raise ModelTrainingError(
            f"Model export failed: {e!s}",
            details={"error": str(e)},
        ) from e


if __name__ == "__main__":
    export_approved_model()
