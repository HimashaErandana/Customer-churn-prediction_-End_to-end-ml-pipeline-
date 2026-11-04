"""Feature engineering module for preparing features for model training."""

import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import FeatureEngineeringError


class FeatureBuilder(PipelineStage):
    """Build feature matrix (X) and target vector (y) for model training.

    This class separates features from the target variable and prepares
    the data for model training. No I/O operations are performed.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the FeatureBuilder.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and target vector.

        Args:
            df: Transformed DataFrame containing features and target.

        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target vector.

        Raises:
            FeatureEngineeringError: If feature engineering fails.
        """
        try:
            self.logger.info(
                "Starting feature engineering",
                rows=len(df),
                columns=list(df.columns),
            )

            X, y = self._split_features_target(df)

            self.logger.info(
                "Feature engineering completed",
                features_shape=X.shape,
                target_shape=y.shape,
            )

            return X, y

        except FeatureEngineeringError:
            raise
        except Exception as e:
            raise FeatureEngineeringError(
                f"Feature engineering failed: {e!s}",
                details={"error": str(e)},
            ) from e

    def _split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features (X) and target (y).

        Args:
            df: DataFrame to split.

        Returns:
            Tuple of (X, y) where X is features and y is target.

        Raises:
            FeatureEngineeringError: If splitting fails.
        """
        try:
            if "Churn" not in df.columns:
                raise FeatureEngineeringError(
                    "Target column 'Churn' not found in DataFrame",
                    details={"columns": list(df.columns)},
                )

            X = df.drop(columns=["Churn"])
            y = df["Churn"]

            self.logger.info(
                "Split features and target",
                features=list(X.columns),
                target_distribution=y.value_counts().to_dict(),
            )

            return X, y

        except FeatureEngineeringError:
            raise
        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to split features and target: {e!s}",
                details={"error": str(e)},
            ) from e
