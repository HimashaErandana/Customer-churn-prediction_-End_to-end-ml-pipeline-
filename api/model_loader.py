"""Model loader for serving the bundled deployment artifact.

Loads the approved model artifact produced by ``src/deployment/export_model.py``.
MLflow is NOT used at serving time — the model, label encoders, and metadata
are bundled directly in the artifact.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]

from src.utils.exceptions import ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEPLOYMENT_MODEL_PATH = Path("deployment/model/model.pkl")


class ModelLoader:
    """Load the approved model from the local deployment artifact.

    The artifact is produced locally by ``src/deployment/export_model.py``
    and contains the model, label encoders, feature names, and metadata.
    No MLflow registry lookup is performed.
    """

    def __init__(self) -> None:
        """Initialize the ModelLoader."""
        self._model: Any = None
        self._model_version: str | None = None
        self._model_info: dict[str, Any] | None = None
        self._label_encoders: dict[str, Any] | None = None
        self._feature_names: list[str] | None = None

    def load_production_model(self) -> Any:
        """Load the model from the local deployment artifact.

        Returns:
            The loaded model.

        Raises:
            ModelLoadError: If the artifact cannot be loaded.
        """
        try:
            model = self._load_from_local_file()
            if model is not None:
                return model
        except Exception as e:
            logger.error("Local model load failed", error=str(e))
            raise ModelLoadError(
                f"Failed to load model from deployment artifact: {e!s}",
                details={"error": str(e)},
            ) from e

        raise ModelLoadError(
            "No model artifact available",
            details={"path": str(DEPLOYMENT_MODEL_PATH)},
        )

    def _load_from_local_file(self) -> Any | None:
        """Load model from the deployment artifact.

        Returns:
            Loaded model, or None if the artifact does not exist.
        """
        try:
            if not DEPLOYMENT_MODEL_PATH.exists():
                logger.warning(
                    "Deployment model artifact not found",
                    path=str(DEPLOYMENT_MODEL_PATH),
                )
                return None

            model_data = joblib.load(DEPLOYMENT_MODEL_PATH)

            self._model = model_data["model"]
            self._label_encoders = model_data.get("label_encoders", {})
            self._feature_names = model_data.get("feature_names", [])

            metadata = model_data.get("metadata", {})
            self._model_version = metadata.get("best_experiment", "local")
            gate_metrics = metadata.get("gate_metrics", {})

            trained_at = datetime.now(tz=UTC).isoformat()
            self._model_info = {
                "version": self._model_version,
                "trained_at": trained_at,
                "metrics": gate_metrics,
                "experiment": metadata.get("best_experiment"),
            }

            logger.info(
                "Model loaded from deployment artifact",
                path=str(DEPLOYMENT_MODEL_PATH),
                version=self._model_version,
                has_encoders=bool(self._label_encoders),
                feature_count=len(self._feature_names),
            )

            return self._model

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from deployment artifact: {e!s}",
                details={"error": str(e)},
            ) from e

    def get_model(self) -> Any:
        """Get the cached model, loading if necessary.

        Returns:
            The loaded model.

        Raises:
            ModelLoadError: If no model can be loaded.
        """
        if self._model is None:
            self.load_production_model()
        return self._model

    def get_model_version(self) -> str:
        """Get the current model version.

        Returns:
            Model version string.
        """
        return self._model_version or "unknown"

    def get_model_info(self) -> dict[str, Any]:
        """Get version, training time and metrics for the loaded model.

        Returns:
            Mapping with version, trained_at, metrics, experiment.

        Raises:
            ModelLoadError: If no model can be loaded or metadata is missing.
        """
        if self._model is None:
            self.load_production_model()
        if self._model_info is None:
            raise ModelLoadError(
                "Model metadata unavailable",
                details={"model_version": self.get_model_version()},
            )
        return self._model_info

    def get_label_encoders(self) -> dict[str, Any]:
        """Get the label encoders used during training.

        Returns:
            Dictionary of column name to LabelEncoder.
        """
        if self._label_encoders is None and self._model is None:
            self.load_production_model()
        return self._label_encoders or {}

    def get_feature_names(self) -> list[str]:
        """Get the feature names used during training.

        Returns:
            List of feature column names.
        """
        if self._feature_names is None and self._model is None:
            self.load_production_model()
        return self._feature_names or []


model_loader = ModelLoader()
