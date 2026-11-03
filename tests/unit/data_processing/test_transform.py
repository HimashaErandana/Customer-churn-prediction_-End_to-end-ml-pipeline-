"""Unit tests for data transformation module."""

import pandas as pd

from src.data_processing.transform import DataTransformer


class TestDataTransformer:
    """Tests for DataTransformer class."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create a test DataFrame for transformation."""
        return pd.DataFrame(
            {
                "customerID": ["001", "002", "003"],
                "gender": ["Male", "Female", "Male"],
                "SeniorCitizen": [0, 1, 0],
                "Partner": ["Yes", "No", "Yes"],
                "Dependents": ["No", "Yes", "No"],
                "tenure": [12, 24, 36],
                "PhoneService": ["Yes", "Yes", "No"],
                "MultipleLines": ["Yes", "No", "No phone service"],
                "InternetService": ["DSL", "Fiber optic", "No"],
                "OnlineSecurity": ["Yes", "No", "No internet service"],
                "OnlineBackup": ["No", "Yes", "No internet service"],
                "DeviceProtection": ["Yes", "No", "No internet service"],
                "TechSupport": ["No", "Yes", "No internet service"],
                "StreamingTV": ["Yes", "No", "No internet service"],
                "StreamingMovies": ["No", "Yes", "No internet service"],
                "Contract": ["Month-to-month", "One year", "Two year"],
                "PaperlessBilling": ["Yes", "No", "Yes"],
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                ],
                "MonthlyCharges": [29.85, 56.95, 42.3],
                "TotalCharges": [29.85, 1889.5, 1840.75],
                "Churn": [0, 1, 0],
            }
        )

    def test_transform_success(self) -> None:
        """Test successful data transformation."""
        df = self._create_test_df()
        transformer = DataTransformer()
        result = transformer.run(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "customerID" not in result.columns

    def test_transform_drops_customer_id(self) -> None:
        """Test that customerID column is dropped."""
        df = self._create_test_df()
        transformer = DataTransformer()
        result = transformer.run(df)

        assert "customerID" not in result.columns

    def test_transform_encodes_categoricals(self) -> None:
        """Test that categorical variables are encoded."""
        df = self._create_test_df()
        transformer = DataTransformer()
        result = transformer.run(df)

        for col in result.select_dtypes(include=["object"]).columns:
            assert result[col].dtype != "object"
