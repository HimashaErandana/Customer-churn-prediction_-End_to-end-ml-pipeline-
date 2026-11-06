# MLOps Production Pipeline

End-to-end machine learning pipeline with MLOps practices, local MLflow experiment tracking, Docker containerization, GitHub Actions CI/CD, and AWS deployment readiness.

## Current Status: Complete

All 8 phases are complete:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Foundations (config, schemas, logging, exceptions) | Done |
| 2 | Data stages (ingest → validate → clean → transform → features) | Done |
| 3 | Training + evaluation (5 XGBoost experiments, MLflow tracking) | Done |
| 4 | Airflow DAG (batch orchestration, parquet handoffs) | Done |
| 5 | FastAPI serving API | Done |
| 6 | Docker (local + production configs) | Done |
| 7 | GitHub Actions CI/CD | Done |
| 8 | AWS deployment preparation (Dockerfile.prod, CI/CD ready) | Done |

## Repository Structure

```
mlops-production-pipeline/
├── README.md
├── ARCHITECTURE.md          # Single source of truth for structure & conventions
├── requirements.txt         # Runtime dependencies
├── pyproject.toml           # Tool config (black, ruff, isort, mypy, pytest)
├── Dockerfile               # Local API image
├── Dockerfile.prod          # Production (gunicorn) image for AWS
├── docker-compose.yml       # Local: API + MLflow tracking server
├── docker-compose.prod.yml  # Production: API only
├── .dockerignore
├── .gitignore
├── .env                     # Local secrets (gitignored, never committed)
│
├── .github/workflows/ci-cd.yml   # CI/CD: lint → type-check → test → verify-model → build → push → deploy
│
├── airflow/dags/ml_pipeline_dag.py   # Batch orchestration (7 stages, parquet handoffs)
│
├── data/{raw,processed,external}/    # Gitignored except .gitkeep
│
├── src/
│   ├── config/
│   │   ├── settings.py          # pydantic BaseSettings (env vars)
│   │   └── config.yaml          # Hyperparameters, experiments, thresholds, paths
│   ├── schemas/models.py        # Pydantic data contracts
│   ├── pipeline/base.py         # PipelineStage abstract base class
│   ├── data_ingestion/extract.py
│   ├── validation/validate.py
│   ├── data_processing/{clean.py, transform.py}
│   ├── features/feature_engineering.py
│   ├── training/{train.py, tune.py, experiment_runner.py}
│   ├── evaluation/evaluate.py
│   └── utils/{logger.py, exceptions.py}
│
├── deployment/
│   ├── model/model.pkl          # Approved deployment artifact (tracked in git)
│   └── aws/                     # AWS deployment documentation
│
├── api/
│   ├── main.py                  # FastAPI app + routes (/health, /model/info, /predict, /predict/batch)
│   ├── schemas.py               # Request/response models
│   ├── model_loader.py          # Loads bundled deployment artifact (NO MLflow dependency at serving)
│   └── exception_handlers.py    # Maps MLPipelineError subtypes to HTTP status codes
│
├── tests/
│   ├── unit/                    # 50 tests, mirrors src/ structure
│   └── integration/             # Placeholder (.gitkeep)
│
├── mlruns/                      # Local MLflow tracking (gitignored)
├── mlartifacts/                 # MLflow artifact store (gitignored)
└── model/                       # Local cache of latest trained model (gitignored)
```

## Architecture

### Training (local only)
```
python run_pipeline.py
```
1. **Ingest → Validate → Clean → Transform → Features** — 5 stages producing parquet handoffs
2. **ExperimentRunner** — trains all 5 XGBoost experiments from `config.yaml`, logs each to its own MLflow experiment under `./mlruns`
3. **Model selection** — picks best experiment by validation `roc_auc`
4. **Evaluation** — gates on untouched holdout against thresholds in `config.yaml`
5. **Export** — if evaluation passes, `export_approved_model()` bundles model + label encoders + feature names + metadata to `deployment/model/model.pkl`

**MLflow is local-only** — used for experiment tracking, hyperparameter comparison, and model selection. Nothing is registered to the MLflow Production stage.

### Serving (FastAPI)
The API loads the bundled deployment artifact directly — **no MLflow dependency at serving time**:
```
docker-compose up          # Local (API + MLflow for tracking only)
docker-compose -f docker-compose.prod.yml up   # Production (API only)
```

Endpoints:
- `GET /health` — Liveness probe
- `GET /model/info` — Model version, trained_at, metrics
- `POST /predict` — Single record prediction
- `POST /predict/batch` — Batch predictions (max 500 records)

Model is loaded **once at startup** via FastAPI lifespan handler, not per-request.

### CI/CD (GitHub Actions)
Triggers on push/PR to `master`:

1. **lint** — ruff, black --check, isort --check
2. **type-check** — mypy --strict
3. **test** — pytest with 80% coverage minimum
4. **verify-model** — checks `deployment/model/model.pkl` exists and is valid (stops deployment if missing/invalid)
5. **build-and-push** — builds `Dockerfile.prod`, pushes to Amazon ECR
6. **deploy** — SSH to EC2, pulls image, restarts container, health check

**Training never runs in CI/CD.** The approved model artifact is exported locally, committed to git, and baked into the Docker image.

## Quick Start

### Local development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training pipeline locally
python run_pipeline.py

# 3. Export the approved model artifact (auto-run if evaluation passes)
python -m src.deployment.export_model

# 4. Start the API locally
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Test it
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```

### Docker (local)
```bash
docker-compose up
```

### Docker (production)
```bash
docker-compose -f docker-compose.prod.yml up
```

## Required Configuration

Create `.env` locally (never committed):
```
ENVIRONMENT=local
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=churn-prediction/xgboost
```

For CI/CD, configure these GitHub Actions secrets:
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — ECR access
- `EC2_HOST`, `EC2_SSH_KEY` — EC2 deployment
- `AWS_REGION` — ECR/EC2 region

## Testing

```bash
pytest tests/unit/ --cov=src --cov=api --cov-fail-under=80
```

50 unit tests covering all pipeline stages, API endpoints, model loader, and exception handlers.

## Linting & Type Checking

```bash
ruff check src/ api/ run_pipeline.py
black --check src/ api/ run_pipeline.py
isort --check-only src/ api/ run_pipeline.py
mypy --strict src/ api/
```

All enforced in CI before any build or deployment.