"""Run the full ML pipeline without Airflow."""

import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.pipeline_config import load_pipeline_config
from src.data_ingestion.extract import DataIngestor
from src.data_processing.clean import DataCleaner
from src.data_processing.transform import DataTransformer
from src.evaluation.evaluate import ModelEvaluator
from src.training.experiment_runner import ExperimentRunner
from src.features.feature_engineering import FeatureBuilder
from src.utils.logger import get_logger, setup_logging
from src.validation.validate import DataValidator

RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

logger = get_logger("run_pipeline")


def run_pipeline() -> None:
    """Run the complete ML pipeline."""
    config = load_pipeline_config()
    setup_logging(level=config.logging.level)

    logger.info("ml_pipeline_started", project=config.project_name)

    # Step 1: Ingest
    logger.info("stage_started", stage="ingest", step="1/7")
    ingestor = DataIngestor(config)
    df = ingestor.run(str(RAW_DATA_PATH))
    logger.info("stage_completed", stage="ingest", rows=df.shape[0], columns=df.shape[1])

    # Step 2: Validate
    logger.info("stage_started", stage="validate", step="2/7")
    validator = DataValidator(config)
    df = validator.run(df)
    logger.info("stage_completed", stage="validate")

    # Step 3: Clean
    logger.info("stage_started", stage="clean", step="3/7")
    cleaner = DataCleaner(config)
    df = cleaner.run(df)
    logger.info("stage_completed", stage="clean", rows=df.shape[0])

    # Step 4: Transform
    logger.info("stage_started", stage="transform", step="4/7")
    transformer = DataTransformer(config)
    df = transformer.run(df)
    logger.info("stage_completed", stage="transform", columns=df.shape[1])

    # Step 5: Build features
    logger.info("stage_started", stage="features", step="5/7")
    builder = FeatureBuilder(config)
    X, y = builder.run(df)
    logger.info("stage_completed", stage="features", x_shape=X.shape, y_shape=y.shape)

    # Split once: the trainer fits on train only, the evaluator gates on the
    # untouched holdout. Config-driven so test_size/random_state live in one place.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )
    logger.info(
        "data_split_completed",
        train_rows=len(X_train),
        test_rows=len(X_test),
        test_size=config.data.test_size,
    )

    # Step 6: Train + experiment
    logger.info("stage_started", stage="train", step="6/7")
    runner = ExperimentRunner(config)
    experiment_results = runner.run(X_train, y_train)

    # Pick the best experiment by validation roc_auc
    best = max(experiment_results, key=lambda r: r["metrics"]["roc_auc"])
    model = best["model"]
    train_metrics = best["metrics"]
    print(f"Best model chosen: {best['experiment_name']} with roc_auc={best['metrics']['roc_auc']:.4f}")
    logger.info(
        "stage_completed",
        stage="train",
        best_experiment=best["experiment_name"],
        train_metrics=train_metrics,
    )

    # Step 7: Evaluate (gate on the holdout the trainer never saw)
    logger.info("stage_started", stage="evaluate", step="7/7")
    evaluator = ModelEvaluator(config)
    passed = evaluator.run(model, X_test, y_test)
    gate_metrics = evaluator.last_metrics

    logger.info(
        "ml_pipeline_results",
        metrics=gate_metrics,
        passed_evaluation=passed,
    )

    # Save model
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.pkl"

    model_data = {
        "model": model,
        "metrics": gate_metrics,
        "train_metrics": train_metrics,
        "best_experiment": best["experiment_name"],
        "best_run_id": best["run_id"],
        "feature_names": X.columns.tolist(),
        "label_encoders": transformer.label_encoders,
        "passed_evaluation": passed,
    }
    joblib.dump(model_data, model_path)
    logger.info("model_saved", path=str(model_path))

    logger.info("ml_pipeline_completed", passed_evaluation=passed)


if __name__ == "__main__":
    run_pipeline()
