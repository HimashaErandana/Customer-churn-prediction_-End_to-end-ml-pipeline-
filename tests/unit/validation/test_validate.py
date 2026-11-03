"""Unit tests for data validation module."""

import pandas as pd
import pytest

from src.utils.exceptions import DataValidationError
from src.validation.validate import DataValidator


class TestDataValidator:
    """Tests for DataValidator class."""

    def _create_valid_df(self) -> pd.DataFrame:
        """Create a valid test DataFrame."""
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
                "TotalCharges": ["29.85", "1889.5", "1840.75"],
                "Churn": ["No", "Yes", "No"],
            }
        )

    def test_validation_success(self) -> None:
        """Test successful validation."""
        df = self._create_valid_df()
        validator = DataValidator()
        result = validator.run(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_validation_missing_columns(self) -> None:
        """Test validation fails with missing columns."""
        df = pd.DataFrame({"customerID": ["001"], "gender": ["Male"]})
        validator = DataValidator()

        with pytest.raises(DataValidationError) as exc_info:
            validator.run(df)

        assert "Missing required columns" in str(exc_info.value)

    def test_validation_duplicate_ids(self) -> None:
        """Test validation fails with duplicate customer IDs."""
        df = self._create_valid_df()
        df.loc[1, "customerID"] = "001"  # Create duplicate
        validator = DataValidator()

        with pytest.raises(DataValidationError) as exc_info:
            validator.run(df)

        assert "duplicate customer IDs" in str(exc_info.value)

    def test_validation_invalid_target(self) -> None:
        """Test validation fails with invalid target values."""
        df = self._create_valid_df()
        df.loc[0, "Churn"] = "Invalid"
        validator = DataValidator()

        with pytest.raises(DataValidationError) as exc_info:
            validator.run(df)

        assert "Invalid values in Churn column" in str(exc_info.value)
