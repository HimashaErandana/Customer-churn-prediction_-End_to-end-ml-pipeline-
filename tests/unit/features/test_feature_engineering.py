"""Unit tests for feature engineering module."""

import pandas as pd
import pytest

from src.features.feature_engineering import FeatureBuilder
from src.utils.exceptions import FeatureEngineeringError


class TestFeatureBuilder:
    """Tests for FeatureBuilder class."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create a test DataFrame for feature engineering."""
        return pd.DataFrame(
            {
                "customerID": ["001", "002", "003"],
                "gender": [0, 1, 0],
                "SeniorCitizen": [0, 1, 0],
                "Partner": [1, 0, 1],
                "Dependents": [0, 1, 0],
                "tenure": [12, 24, 36],
                "PhoneService": [1, 1, 0],
                "MultipleLines": [1, 0, 2],
                "InternetService": [0, 1, 2],
                "OnlineSecurity": [1, 0, 2],
                "OnlineBackup": [0, 1, 2],
                "DeviceProtection": [1, 0, 2],
                "TechSupport": [0, 1, 2],
                "StreamingTV": [1, 0, 2],
                "StreamingMovies": [0, 1, 2],
                "Contract": [0, 1, 2],
                "PaperlessBilling": [1, 0, 1],
                "PaymentMethod": [0, 1, 2],
                "MonthlyCharges": [29.85, 56.95, 42.3],
                "TotalCharges": [29.85, 1889.5, 1840.75],
                "Churn": [0, 1, 0],
            }
        )

    def test_feature_engineering_success(self) -> None:
        """Test successful feature engineering."""
        df = self._create_test_df()
        builder = FeatureBuilder()
        X, y = builder.run(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 3
        assert len(y) == 3
        assert "Churn" not in X.columns

    def test_feature_engineering_missing_target(self) -> None:
        """Test error handling when target column is missing."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = FeatureBuilder()

        with pytest.raises(FeatureEngineeringError) as exc_info:
            builder.run(df)

        assert "Target column 'Churn' not found" in str(exc_info.value)

    def test_feature_engineering_returns_correct_shapes(self) -> None:
        """Test that feature engineering returns correct shapes."""
        df = self._create_test_df()
        builder = FeatureBuilder()
        X, y = builder.run(df)

        assert X.shape == (3, 20)
        assert y.shape == (3,)
