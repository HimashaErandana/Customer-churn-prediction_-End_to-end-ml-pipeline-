"""Model loader for fetching model from MLflow Registry or local file."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.artifacts
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from src.config.settings import settings
from src.utils.exceptions import ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_MODEL_PATH = Path("model/model.pkl")


class ModelLoader:
    """Load the stage model from the MLflow Registry or a local file.

    Tries the registry first (settings.MODEL_NAME at settings.MODEL_STAGE);
    falls back to the local model.pkl cache when the registry has nothing.
    """

    def __init__(self) -> None:
        """Initialize the ModelLoader."""
        self._model: Any = None
        self._model_version: str | None = None
        self._model_info: dict[str, Any] | None = None
        self._model_data: dict[str, Any] | None = None
        self._label_encoders: dict[str, Any] | None = None

    def load_production_model(self) -> Any:
        """Fetch the model from the registry, falling back to the local file.

        Returns:
            The loaded model.

        Raises:
            ModelLoadError: If neither source has a model.
        """
        try:
            model = self._load_from_registry()
            if model is not None:
                return model
        except Exception as e:
            logger.warning("Registry load failed, trying local file", error=str(e))

        return self._load_from_local_file()

    def _load_from_registry(self) -> Any | None:
        """Load the model at settings.MODEL_STAGE from the MLflow Registry.

        Returns:
            Loaded model, or None if the registry has no version in stage.
        """
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.get_latest_versions(settings.MODEL_NAME, stages=[settings.MODEL_STAGE])
        if not versions:
            logger.warning(
                "No model version in stage",
                model_name=settings.MODEL_NAME,
                stage=settings.MODEL_STAGE,
            )
            return None

        mv = versions[0]
        model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
        model = mlflow.xgboost.load_model(model_uri)

        self._model = model
        self._model_version = f"v{mv.version}"
        self._label_encoders = self._load_encoders(client, mv.run_id)
        self._model_info = self._registry_info(client, mv)

        logger.info(
            "Model loaded from registry",
            model_name=settings.MODEL_NAME,
            version=mv.version,
            stage=settings.MODEL_STAGE,
            has_encoders=bool(self._label_encoders),
        )
        return model

    def _load_encoders(self, client: MlflowClient, run_id: str | None) -> dict[str, Any]:
        """Download the label encoders artifact logged with the training run.

        Args:
            client: MLflow client.
            run_id: Run that produced the model version.

        Returns:
            Mapping of column name to LabelEncoder, empty if unavailable.
        """
        if not run_id:
            logger.warning("Model version has no run_id, encoders unavailable")
            return {}
        try:
            local_path = mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/encoders/label_encoders.joblib"
            )
            encoders: dict[str, Any] = joblib.load(local_path)
            logger.info("Encoders loaded from registry", run_id=run_id)
            return encoders
        except Exception as e:
            logger.warning(
                "Encoders artifact not found in run",
                run_id=run_id,
                error=str(e),
            )
            return {}

    @staticmethod
    def _registry_info(client: MlflowClient, mv: Any) -> dict[str, Any]:
        """Build metadata for a registry model version.

        Args:
            client: MLflow client.
            mv: ModelVersion retrieved from the registry.

        Returns:
            Mapping with version, trained_at and metrics.
        """
        trained_at = "unknown"
        metrics: dict[str, float] = {}
        if mv.run_id:
            try:
                run = client.get_run(mv.run_id)
                metrics = dict(run.data.metrics)
                ts = run.info.end_time or run.info.start_time
                if ts:
                    trained_at = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            except Exception as e:
                logger.warning("Run metadata unavailable", run_id=mv.run_id, error=str(e))
        if trained_at == "unknown" and mv.creation_timestamp:
            trained_at = datetime.fromtimestamp(
                mv.creation_timestamp / 1000, tz=timezone.utc
            ).isoformat()
        return {
            "version": f"v{mv.version}",
            "trained_at": trained_at,
            "metrics": metrics,
        }

    def _load_from_local_file(self) -> Any:
        """Load model from local model.pkl file.

        Returns:
            The loaded model.

        Raises:
            ModelLoadError: If local file cannot be loaded.
        """
        try:
            if not LOCAL_MODEL_PATH.exists():
                raise ModelLoadError(
                    f"Local model file not found: {LOCAL_MODEL_PATH}",
                    details={"path": str(LOCAL_MODEL_PATH)},
                )

            self._model_data = joblib.load(LOCAL_MODEL_PATH)
            self._model = self._model_data["model"]
            self._label_encoders = self._model_data.get("label_encoders", {})
            self._model_version = "local"
            trained_at = datetime.fromtimestamp(
                LOCAL_MODEL_PATH.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            self._model_info = {
                "version": "local",
                "trained_at": trained_at,
                "metrics": self._model_data.get("metrics", {}),
            }

            logger.info(
                "Model loaded from local file",
                path=str(LOCAL_MODEL_PATH),
                metrics=self._model_info["metrics"],
                has_encoders=bool(self._label_encoders),
            )

            return self._model

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from local file: {str(e)}",
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
            Mapping with version (str), trained_at (ISO-8601 str) and
            metrics (dict of metric name to value).

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


model_loader = ModelLoader()
