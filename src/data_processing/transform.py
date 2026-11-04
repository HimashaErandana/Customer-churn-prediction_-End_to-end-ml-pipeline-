"""Data transformation module for encoding and scaling."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-untyped]

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import DataProcessingError


class DataTransformer(PipelineStage):
    """Transform data by encoding categorical variables and scaling numerical features.

    This class performs data transformation without feature creation.
    It encodes categorical variables and scales numerical features.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the DataTransformer.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame.

        Args:
            df: Cleaned DataFrame to transform.

        Returns:
            Transformed DataFrame with encoded and scaled features.

        Raises:
            DataProcessingError: If transformation fails.
        """
        try:
            self.logger.info(
                "Starting data transformation",
                rows=len(df),
                columns=list(df.columns),
            )

            df = self._drop_customer_id(df)
            df = self._encode_categoricals(df)

            self.logger.info(
                "Data transformation completed",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except DataProcessingError:
            raise
        except Exception as e:
            raise DataProcessingError(
                f"Data transformation failed: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _drop_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the customerID column.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame without customerID column.

        Raises:
            DataProcessingError: If dropping fails.
        """
        try:
            df = df.drop(columns=["customerID"])
            self.logger.info("Dropped customerID column")
            return df
        except Exception as e:
            raise DataProcessingError(
                f"Failed to drop customerID: {str(e)}",
                details={"error": str(e)},
            ) from e

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder.

        Args:
            df: DataFrame to process.

        Returns:
            DataFrame with encoded categorical variables.

        Raises:
            DataProcessingError: If encoding fails.
        """
        try:
            categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

            for col in categorical_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                self.logger.debug("Encoded column", column=col, classes=list(le.classes_))

            self.logger.info(
                "Encoded categorical variables",
                columns=categorical_columns,
            )

            return df
        except Exception as e:
            raise DataProcessingError(
                f"Failed to encode categorical variables: {str(e)}",
                details={"error": str(e)},
            ) from e
