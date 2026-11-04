"""Data ingestion module for loading raw data."""

from pathlib import Path

import pandas as pd

from src.config.pipeline_config import PipelineConfig
from src.pipeline.base import PipelineStage
from src.utils.exceptions import DataIngestionError


class DataIngestor(PipelineStage):
    """Ingest raw data from various sources.

    This class is responsible for reading raw data from CSV files
    and returning a pandas DataFrame. No cleaning or transformation
    is performed at this stage.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the DataIngestor.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        super().__init__(config)

    def run(self, source_path: str) -> pd.DataFrame:
        """Load raw customer records from the configured data source.

        Args:
            source_path: Path or URI to the raw data file.

        Returns:
            DataFrame with the raw, unvalidated schema.

        Raises:
            DataIngestionError: If the source cannot be read.
        """
        try:
            self.logger.info("Loading raw data", source_path=source_path)

            path = Path(source_path)
            if not path.exists():
                raise DataIngestionError(
                    f"Data file not found: {source_path}",
                    details={"source_path": source_path},
                )

            df = pd.read_csv(path)
            self.logger.info(
                "Loaded raw data successfully",
                rows=len(df),
                columns=len(df.columns),
            )

            return df

        except DataIngestionError:
            raise
        except Exception as e:
            raise DataIngestionError(
                f"Failed to load data from {source_path}: {e!s}",
                details={"source_path": source_path, "error": str(e)},
            ) from e
