"""Data cleaning module for handling nulls, dtypes, and deduplication."""

import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import DataProcessingError


class DataCleaner(PipelineStage):
    """Clean raw data by handling nulls, dtypes, and deduplication.

    This class performs data cleaning operations without feature creation.
    It ensures data quality by fixing data type issues and removing duplicates.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the DataCleaner.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input DataFrame.

        Args:
            df: Raw DataFrame to clean.

        Returns:
            Cleaned DataFrame with proper dtypes and no duplicates.

        Raises:
            DataProcessingError: If cleaning fails.
        """
        try:
            self.logger.info(
                "Starting data cleaning",
                rows=len(df),
                columns=list(df.columns),
            )

            df = self._convert_total_charges(df)
            df = self._handle_missing_values(df)
            df = self._encode_target(df)

            self.logger.info(
                "Data cleaning completed",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except DataProcessingError:
            raise
        except Exception as e:
            raise DataProcessingError(
                f"Data cleaning failed: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _convert_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert TotalCharges to numeric, handling empty strings.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with TotalCharges converted to float.

        Raises:
            DataProcessingError: If conversion fails.
        """
        try:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            return df
        except Exception as e:
            raise DataProcessingError(
                f"Failed to convert TotalCharges: {str(e)}",
                details={"column": "TotalCharges", "error": str(e)},
            ) from e

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with missing values handled.

        Raises:
            DataProcessingError: If handling fails.
        """
        try:
            missing_before = df.isnull().sum().sum()

            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

            missing_after = df.isnull().sum().sum()

            self.logger.info(
                "Handled missing values",
                missing_before=int(missing_before),
                missing_after=int(missing_after),
            )

            return df
        except Exception as e:
            raise DataProcessingError(
                f"Failed to handle missing values: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the target column (Churn) to binary.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with encoded target column.

        Raises:
            DataProcessingError: If encoding fails.
        """
        try:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

            self.logger.info(
                "Encoded target column",
                distribution=df["Churn"].value_counts().to_dict(),
            )

            return df
        except Exception as e:
            raise DataProcessingError(
                f"Failed to encode target column: {str(e)}",
                details={"column": "Churn", "error": str(e)},
            ) from e
