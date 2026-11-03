"""Data validation module using Great Expectations."""

import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import DataValidationError


class DataValidator(PipelineStage):
    """Validate data quality using Great Expectations.

    This class runs validation checks on the raw data to ensure
    it meets quality standards before processing.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the DataValidator.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Great Expectations suite on the input DataFrame.

        Args:
            df: Raw DataFrame to validate.

        Returns:
            DataFrame unchanged if validation passes.

        Raises:
            DataValidationError: If validation fails.
        """
        try:
            self.logger.info(
                "Starting data validation",
                rows=len(df),
                columns=list(df.columns),
            )

            self._validate_required_columns(df)
            self._validate_data_types(df)
            self._validate_no_duplicates(df)
            self._validate_target_column(df)

            self.logger.info("Data validation passed")
            return df

        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"Validation failed: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present.

        Args:
            df: DataFrame to validate.

        Raises:
            DataValidationError: If required columns are missing.
        """
        required_columns = [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns},
            )

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate that columns have expected data types.

        Args:
            df: DataFrame to validate.

        Raises:
            DataValidationError: If data types are invalid.
        """
        if not pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
            raise DataValidationError(
                "SeniorCitizen must be numeric",
                details={"column": "SeniorCitizen", "dtype": str(df["SeniorCitizen"].dtype)},
            )

        if not pd.api.types.is_numeric_dtype(df["tenure"]):
            raise DataValidationError(
                "tenure must be numeric",
                details={"column": "tenure", "dtype": str(df["tenure"].dtype)},
            )

        if not pd.api.types.is_numeric_dtype(df["MonthlyCharges"]):
            raise DataValidationError(
                "MonthlyCharges must be numeric",
                details={"column": "MonthlyCharges", "dtype": str(df["MonthlyCharges"].dtype)},
            )

    def _validate_no_duplicates(self, df: pd.DataFrame) -> None:
        """Validate that there are no duplicate customer IDs.

        Args:
            df: DataFrame to validate.

        Raises:
            DataValidationError: If duplicates are found.
        """
        duplicate_ids = df["customerID"].duplicated().sum()
        if duplicate_ids > 0:
            raise DataValidationError(
                f"Found {duplicate_ids} duplicate customer IDs",
                details={"duplicate_count": int(duplicate_ids)},
            )

    def _validate_target_column(self, df: pd.DataFrame) -> None:
        """Validate that the target column has valid values.

        Args:
            df: DataFrame to validate.

        Raises:
            DataValidationError: If target column has invalid values.
        """
        valid_values = {"Yes", "No"}
        invalid_values = set(df["Churn"].unique()) - valid_values
        if invalid_values:
            raise DataValidationError(
                f"Invalid values in Churn column: {invalid_values}",
                details={"invalid_values": list(invalid_values)},
            )
