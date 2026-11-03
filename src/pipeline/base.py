"""Abstract base class for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Any

from src.config.pipeline_config import PipelineConfig, load_pipeline_config
from src.utils.logger import get_logger


class PipelineStage(ABC):
    """Base class for every stage the Airflow DAG can call.

    All pipeline stages must inherit from this class and implement the run() method.
    This provides a consistent interface for the Airflow DAG to call stages
    without knowing their internals.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline stage.

        Args:
            config: Pipeline configuration. Loaded from config.yaml when omitted.
        """
        self.config = config if config is not None else load_pipeline_config()
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Execute this stage and return its output.

        Args:
            data: Input data for this stage.

        Returns:
            Output data from this stage.

        Raises:
            MLPipelineError: If the stage fails.
        """
