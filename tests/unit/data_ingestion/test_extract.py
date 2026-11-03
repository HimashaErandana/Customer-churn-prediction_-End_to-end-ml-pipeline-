"""Unit tests for data ingestion module."""

import pandas as pd
import pytest

from src.data_ingestion.extract import DataIngestor
from src.utils.exceptions import DataIngestionError


class TestDataIngestor:
    """Tests for DataIngestor class."""

    def test_load_csv_success(self, tmp_path: object) -> None:
        """Test successful CSV loading."""
        csv_path = tmp_path / "test.csv"  # type: ignore
        df_test = pd.DataFrame(
            {
                "customerID": ["001", "002"],
                "gender": ["Male", "Female"],
                "Churn": ["Yes", "No"],
            }
        )
        df_test.to_csv(csv_path, index=False)

        ingestor = DataIngestor()
        result = ingestor.run(str(csv_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["customerID", "gender", "Churn"]

    def test_load_csv_file_not_found(self) -> None:
        """Test error handling when file does not exist."""
        ingestor = DataIngestor()

        with pytest.raises(DataIngestionError) as exc_info:
            ingestor.run("nonexistent.csv")

        assert "Data file not found" in str(exc_info.value)

    def test_load_csv_invalid_format(self, tmp_path: object) -> None:
        """Test error handling for invalid CSV format."""
        csv_path = tmp_path / "invalid.csv"  # type: ignore
        csv_path.write_text("not,a,valid,csv\n1,2,3")

        ingestor = DataIngestor()
        result = ingestor.run(str(csv_path))

        assert isinstance(result, pd.DataFrame)
