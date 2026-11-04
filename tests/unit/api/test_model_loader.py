"""Unit tests for model_loader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from api.model_loader import DEPLOYMENT_MODEL_PATH, ModelLoader
from src.utils.exceptions import ModelLoadError


class TestModelLoader:
    """Tests for ModelLoader class."""

    def _make_test_artifact(self) -> dict:
        """Create a minimal deployment model artifact dict (not pickled)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        return {
            "model": mock_model,
            "label_encoders": {},
            "feature_names": ["feature1", "feature2"],
            "metadata": {
                "best_experiment": "xgboost_v3",
                "gate_metrics": {"roc_auc": 0.85, "accuracy": 0.8},
                "train_metrics": {"roc_auc": 0.9, "accuracy": 0.85},
            },
        }

    def test_load_production_model_success(self, tmp_path: Path) -> None:
        """Test successful model loading from deployment artifact."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()

        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            model = loader.load_production_model()

        assert model is not None
        assert loader._model is not None
        assert loader._model_version == "xgboost_v3"
        assert loader._model_info is not None
        assert loader._model_info["version"] == "xgboost_v3"
        assert loader._model_info["metrics"]["roc_auc"] == 0.85
        assert loader._feature_names == ["feature1", "feature2"]

    def test_load_production_model_file_not_found(self) -> None:
        """Test model loading when artifact does not exist."""
        loader = ModelLoader()

        with patch("api.model_loader.DEPLOYMENT_MODEL_PATH", Path("/nonexistent/model.pkl")):
            try:
                loader.load_production_model()
                assert False, "Expected ModelLoadError"
            except ModelLoadError as e:
                assert "No model artifact available" in str(e)

    def test_load_from_local_file_exception(self, tmp_path: Path) -> None:
        """Test handling of corrupted model artifact."""
        loader = ModelLoader()
        artifact_path = tmp_path / "corrupted.pkl"
        artifact_path.touch()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", side_effect=Exception("corrupt")),
        ):
            try:
                loader.load_production_model()
                assert False, "Expected ModelLoadError"
            except ModelLoadError as e:
                assert "Failed to load model" in str(e)

    def test_get_model_caching(self, tmp_path: Path) -> None:
        """Test that get_model caches the loaded model."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()
        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            model1 = loader.get_model()
            model2 = loader.get_model()

        assert model1 is model2  # Same cached instance

    def test_get_model_load_error(self) -> None:
        """Test get_model raises when artifact is missing."""
        loader = ModelLoader()

        with patch("api.model_loader.DEPLOYMENT_MODEL_PATH", Path("/nonexistent/model.pkl")):
            try:
                loader.get_model()
                assert False, "Expected ModelLoadError"
            except ModelLoadError:
                pass

    def test_get_model_version_loaded(self, tmp_path: Path) -> None:
        """Test get_model_version when model is loaded."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()
        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            loader.load_production_model()
            version = loader.get_model_version()

        assert version == "xgboost_v3"

    def test_get_model_version_not_loaded(self) -> None:
        """Test get_model_version returns default when no model loaded."""
        loader = ModelLoader()
        assert loader.get_model_version() == "unknown"

    def test_get_model_info_success(self, tmp_path: Path) -> None:
        """Test get_model_info returns metadata when model loaded."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()
        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            loader.load_production_model()
            info = loader.get_model_info()

        assert info["version"] == "xgboost_v3"
        assert "trained_at" in info
        assert info["metrics"]["roc_auc"] == 0.85

    def test_get_model_info_not_loaded(self) -> None:
        """Test get_model_info raises when model not loaded."""
        loader = ModelLoader()

        with patch("api.model_loader.DEPLOYMENT_MODEL_PATH", Path("/nonexistent/model.pkl")):
            try:
                loader.get_model_info()
                assert False, "Expected ModelLoadError"
            except ModelLoadError as e:
                assert "No model artifact available" in str(e)

    def test_get_label_encoders(self, tmp_path: Path) -> None:
        """Test get_label_encoders returns encoders from artifact."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()
        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            loader.load_production_model()
            encoders = loader.get_label_encoders()

        assert encoders == {}

    def test_get_feature_names(self, tmp_path: Path) -> None:
        """Test get_feature_names returns feature names from artifact."""
        artifact = self._make_test_artifact()
        artifact_path = tmp_path / "model.pkl"
        artifact_path.touch()
        loader = ModelLoader()

        with (
            patch("api.model_loader.DEPLOYMENT_MODEL_PATH", artifact_path),
            patch("api.model_loader.joblib.load", return_value=artifact),
        ):
            loader.load_production_model()
            features = loader.get_feature_names()

        assert features == ["feature1", "feature2"]
