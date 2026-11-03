"""Airflow DAG for the ML pipeline.

Data flows between tasks as parquet files under data/processed/; XCom only
carries small path strings, metric dicts, and the MLflow run id.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator

from airflow import DAG

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.pipeline_config import load_pipeline_config
from src.config.settings import settings
from src.data_ingestion.extract import DataIngestor
from src.data_processing.clean import DataCleaner
from src.data_processing.transform import DataTransformer
from src.evaluation.evaluate import ModelEvaluator
from src.features.feature_engineering import FeatureBuilder
from src.training.train import ModelTrainer
from src.utils.logger import get_logger
from src.validation.validate import DataValidator

logger = get_logger(__name__)

PROCESSED_DIR = Path(settings.DATA_PROCESSED_PATH)
RAW_DATA_PATH = Path(settings.DATA_RAW_PATH) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ENCODERS_PATH = PROCESSED_DIR / "label_encoders.joblib"

default_args = {
    "owner": "ml-pipeline",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DAG_ID = "ml_churn_pipeline"


def _read_parquet(path: str) -> pd.DataFrame:
    """Read a parquet handoff file.

    Args:
        path: Path to the parquet file.

    Returns:
        The loaded DataFrame.
    """
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, filename: str) -> str:
    """Write a handoff file under data/processed/.

    Args:
        df: DataFrame to persist.
        filename: File name within the processed directory.

    Returns:
        The absolute path as a string, ready for XCom.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_parquet(out, index=False)
    return str(out)


def ingest_data(**kwargs: object) -> str:
    """Ingest raw data from CSV and persist it.

    Returns:
        Path to the raw handoff file.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    df = DataIngestor(config).run(str(RAW_DATA_PATH))
    path = _write_parquet(df, "01_raw.parquet")
    ti.xcom_push(key="path", value=path)
    return path


def validate_data(**kwargs: object) -> str:
    """Validate the raw handoff file.

    Returns:
        Path to the validated handoff file.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    df = _read_parquet(ti.xcom_pull(key="path", task_ids="ingest_data"))
    validated = DataValidator(config).run(df)
    path = _write_parquet(validated, "02_validated.parquet")
    ti.xcom_push(key="path", value=path)
    return path


def clean_data(**kwargs: object) -> str:
    """Clean the validated handoff file.

    Returns:
        Path to the cleaned handoff file.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    df = _read_parquet(ti.xcom_pull(key="path", task_ids="validate_data"))
    cleaned = DataCleaner(config).run(df)
    path = _write_parquet(cleaned, "03_cleaned.parquet")
    ti.xcom_push(key="path", value=path)
    return path


def transform_data(**kwargs: object) -> str:
    """Transform the cleaned handoff file and persist label encoders.

    Returns:
        Path to the transformed handoff file.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    df = _read_parquet(ti.xcom_pull(key="path", task_ids="clean_data"))
    transformer = DataTransformer(config)
    transformed = transformer.run(df)
    path = _write_parquet(transformed, "04_transformed.parquet")
    joblib.dump(transformer.label_encoders, ENCODERS_PATH)
    ti.xcom_push(key="path", value=path)
    return path


def build_features(**kwargs: object) -> dict[str, str]:
    """Build features and split into train/test holdout files.

    Returns:
        Mapping of split name to parquet path.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    df = _read_parquet(ti.xcom_pull(key="path", task_ids="transform_data"))
    X, y = FeatureBuilder(config).run(df)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    target = config.data.target_column
    paths = {
        "X_train": _write_parquet(X_train, "X_train.parquet"),
        "X_test": _write_parquet(X_test, "X_test.parquet"),
        "y_train": _write_parquet(y_train.to_frame(name=target), "y_train.parquet"),
        "y_test": _write_parquet(y_test.to_frame(name=target), "y_test.parquet"),
    }
    ti.xcom_push(key="paths", value=paths)
    return paths


def train_model(**kwargs: object) -> dict[str, object]:
    """Train the model on the training split.

    Returns:
        Mapping with the MLflow run id and training metrics.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    paths: dict[str, str] = ti.xcom_pull(key="paths", task_ids="build_features")

    X_train = _read_parquet(paths["X_train"])
    y_train = _read_parquet(paths["y_train"])[config.data.target_column]

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    trainer = ModelTrainer(config)
    params = dict(config.model.xgboost)
    _, train_metrics = trainer.run(X_train, y_train, params)

    run_id = mlflow.last_active_run().info.run_id
    result = {"run_id": run_id, "train_metrics": train_metrics}
    ti.xcom_push(key="train_result", value=result)
    return result


def evaluate_model(**kwargs: object) -> bool:
    """Gate the model on the untouched test split.

    Returns:
        True if all promotion thresholds are met.
    """
    ti = kwargs["ti"]
    config = load_pipeline_config()
    train_result: dict[str, object] = ti.xcom_pull(key="train_result", task_ids="train_model")
    paths: dict[str, str] = ti.xcom_pull(key="paths", task_ids="build_features")

    X_test = _read_parquet(paths["X_test"])
    y_test = _read_parquet(paths["y_test"])[config.data.target_column]

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model(f"runs:/{train_result['run_id']}/model")

    passed = ModelEvaluator(config).run(model, X_test, y_test)
    ti.xcom_push(key="evaluation_passed", value=passed)
    return passed


def branch_on_evaluation(**kwargs: object) -> str:
    """Branch based on evaluation result.

    Returns:
        Task id to execute next.
    """
    ti = kwargs["ti"]
    passed = ti.xcom_pull(key="evaluation_passed", task_ids="evaluate_model")
    if passed:
        return "register_model"
    return "notify_failure"


def register_model(**kwargs: object) -> None:
    """Register the model in MLflow and promote it to Staging."""
    from mlflow.tracking import MlflowClient

    ti = kwargs["ti"]
    train_result: dict[str, object] = ti.xcom_pull(key="train_result", task_ids="train_model")
    run_id = str(train_result["run_id"])

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Label encoders ride along in the run so serving can decode categoricals.
    if ENCODERS_PATH.exists():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(ENCODERS_PATH), artifact_path="encoders")

    mv = mlflow.register_model(f"runs:/{run_id}/model", settings.MODEL_NAME)
    client = MlflowClient()
    client.transition_model_version_stage(
        settings.MODEL_NAME,
        mv.version,
        "Staging",
        archive_existing_versions=True,
    )
    logger.info(
        "model_registered",
        model_name=settings.MODEL_NAME,
        version=mv.version,
        run_id=run_id,
        stage="Staging",
    )


def notify_failure(**kwargs: object) -> None:
    """Record that the evaluation gate was not met."""
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(key="train_result", task_ids="train_model")
    logger.warning("model_evaluation_failed", train_result=metrics)


with DAG(
    DAG_ID,
    default_args=default_args,
    description="End-to-end ML pipeline for customer churn prediction",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "churn", "xgboost"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    features = PythonOperator(
        task_id="build_features",
        python_callable=build_features,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_evaluation",
        python_callable=branch_on_evaluation,
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    failure = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure,
    )

    (
        ingest
        >> validate
        >> clean
        >> transform
        >> features
        >> train
        >> evaluate
        >> branch
        >> [register, failure]
    )
