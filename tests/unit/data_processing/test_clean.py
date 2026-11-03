"""Unit tests for data cleaning module."""

import pandas as pd

from src.data_processing.clean import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner class."""

    def _create_test_df(self) -> pd.DataFrame:
        """Create a test DataFrame with cleaning issues."""
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
                "TotalCharges": ["29.85", "", "1840.75"],
                "Churn": ["No", "Yes", "No"],
            }
        )

    def test_clean_success(self) -> None:
        """Test successful data cleaning."""
        df = self._create_test_df()
        cleaner = DataCleaner()
        result = cleaner.run(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result["Churn"].dtype in ["int64", "int32"]
        assert result["TotalCharges"].dtype == "float64"

    def test_clean_handles_missing_total_charges(self) -> None:
        """Test that missing TotalCharges values are handled."""
        df = self._create_test_df()
        cleaner = DataCleaner()
        result = cleaner.run(df)

        assert not result["TotalCharges"].isna().any()

    def test_clean_encodes_target(self) -> None:
        """Test that target column is encoded to binary."""
        df = self._create_test_df()
        cleaner = DataCleaner()
        result = cleaner.run(df)

        assert set(result["Churn"].unique()).issubset({0, 1})
