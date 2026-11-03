# Architecture & Development Standards
### Production ML Pipeline with MLOps and AWS Deployment

> **Purpose of this document.** This is the single source of truth for how this project is
> structured, named, and built. It exists so a coding agent (or a human) can implement any
> module in isolation and have it fit perfectly with every other module — no guessing at
> function signatures, no inventing folder structures, no ad-hoc logging styles.
>
> **Rule for agents:** if something in a task is not covered here, do not improvise a new
> convention — add it to this document first, then implement it.

Example domain used throughout for illustration: **customer churn prediction** (binary
classification, tabular data). Replace with your real use case — the contracts/shapes stay
the same.

---

## 1. System Architecture

```
                         DATA SOURCES
                    (CSV / API / Database)
                              │
                              ▼
                    ┌──────────────────────┐
                    │ 1. Ingestion          │  src/data_ingestion/
                    │    Airflow task       │
                    └──────────────────────┘
                              │  raw DataFrame
                              ▼
                    ┌──────────────────────┐
                    │ 2. Validation         │  src/validation/
                    │    Great Expectations │
                    └──────────────────────┘
                              │  validated DataFrame (or hard fail)
                              ▼
                    ┌──────────────────────┐
                    │ 3. Cleaning/Transform │  src/data_processing/
                    └──────────────────────┘
                              │  processed DataFrame
                              ▼
                    ┌──────────────────────┐
                    │ 4. Feature Engineering│  src/features/
                    └──────────────────────┘
                              │  feature matrix (X), target (y)
                              ▼
                    ┌──────────────────────┐
                    │ 5. Training + Tuning  │  src/training/
                    │    logged to MLflow   │
                    └──────────────────────┘
                              │  model artifact + metrics
                              ▼
                    ┌──────────────────────┐
                    │ 6. Evaluation         │  src/evaluation/
                    │    gate: promote?     │
                    └──────────────────────┘
                              │  pass → register in MLflow Model Registry
                              ▼
                    ┌──────────────────────┐
                    │ 7. FastAPI serving    │  api/
                    │    loads Production   │
                    │    stage model        │
                    └──────────────────────┘
                              │
                              ▼
                    Docker image → GitHub Actions CI/CD → ECR → EC2
```

**Control flow vs data flow, explicitly:**
- Steps 1–6 run inside **Airflow** (batch, scheduled or triggered).
- Step 7 is a **long-running service**, independent of Airflow, that only ever reads the
  model MLflow has marked `Production` — it never triggers training.
- The only thing that crosses the batch/serving boundary is the registered model + its
  signature (input schema). This boundary is a hard rule: **the API never imports anything
  from `src/training/`.**

---

## 2. Repository Structure

```
mlops-production-pipeline/
├── README.md
├── requirements.txt
├── requirements-dev.txt        # black, ruff, mypy, pytest, pre-commit
├── pyproject.toml              # tool config: black, ruff, mypy, pytest
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── .env.example                # documents every required env var, no real secrets
│
├── .github/workflows/ci-cd.yml
│
├── airflow/dags/ml_pipeline_dag.py
│
├── data/{raw,processed,external}/   # gitignored except .gitkeep
│
├── src/
│   ├── config/
│   │   ├── settings.py          # pydantic BaseSettings (env vars)
│   │   └── config.yaml          # hyperparameters, thresholds, paths
│   ├── schemas/
│   │   └── models.py            # all Pydantic data contracts (Section 5)
│   ├── pipeline/
│   │   └── base.py              # PipelineStage abstract base class (Section 3.9)
│   ├── data_ingestion/extract.py
│   ├── validation/validate.py
│   ├── data_processing/{clean.py,transform.py}
│   ├── features/feature_engineering.py
│   ├── training/{train.py,tune.py}
│   ├── evaluation/evaluate.py
│   └── utils/{logger.py,exceptions.py}
│
├── model/                       # local cache of latest Production model (gitignored)
│
├── api/
│   ├── main.py                  # FastAPI app + routes
│   ├── schemas.py                # request/response models (imports src/schemas)
│   ├── model_loader.py           # loads model from MLflow Registry
│   └── exception_handlers.py
│
├── tests/
│   ├── unit/                     # mirrors src/ structure 1:1
│   └── integration/              # API + pipeline end-to-end tests
│
└── deployment/
    ├── aws/{ecr_setup.md,ec2_deployment.md}
    └── nginx/nginx.conf
```

**Rule:** every file under `tests/unit/` mirrors a file under `src/` with the same relative
path and a `test_` prefix (e.g. `src/features/feature_engineering.py` →
`tests/unit/features/test_feature_engineering.py`). An agent implementing a module writes
its test file in the same task.

---

## 3. Coding Standards

### 3.1 Formatting & linting (non-negotiable, enforced by CI)
- **Formatter:** `black`, line length 100.
- **Import sorting:** `isort` (black-compatible profile).
- **Linter:** `ruff` (replaces flake8), rules: `E, F, I, B, UP`.
- **Type checking:** `mypy --strict` on `src/` and `api/`. Every function has full type
  hints — parameters and return type, no exceptions.
- All four run in a `pre-commit` hook and again as a CI gate.

### 3.2 Naming conventions
| Element | Convention | Example |
|---|---|---|
| Files/modules | `snake_case` | `feature_engineering.py` |
| Classes | `PascalCase` | `ChurnPredictor` |
| Functions/variables | `snake_case` | `load_raw_data()` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TEST_SIZE` |
| Pydantic schemas | `PascalCase` + noun | `PredictionRequest`, `RawCustomerRecord` |
| Env vars | `UPPER_SNAKE_CASE` | `MLFLOW_TRACKING_URI` |
| Branches | `feature/…`, `fix/…`, `chore/…` | `feature/add-shap-explanations` |
| Commits | [Conventional Commits](https://www.conventionalcommits.org) | `feat(training): add xgboost tuning` |

### 3.3 Docstrings
Google style, mandatory on every public function/class:
```python
def load_raw_data(source_path: str) -> pd.DataFrame:
    """Load raw customer records from the configured data source.

    Args:
        source_path: Path or URI to the raw data file.

    Returns:
        DataFrame with the raw, unvalidated schema (see RawCustomerRecord).

    Raises:
        DataIngestionError: If the source cannot be read.
    """
```

### 3.4 Logging
- Never use `print()`. Always use the shared logger from `src/utils/logger.py`.
- One logger per module: `logger = get_logger(__name__)`.
- Structured (JSON) output in production, human-readable in local dev — controlled by
  `settings.ENVIRONMENT`.
- Levels: `DEBUG` (dev detail), `INFO` (pipeline milestones, e.g. "loaded 10,432 rows"),
  `WARNING` (recoverable data issues), `ERROR` (operation failed), `CRITICAL` (service down).

### 3.5 Error handling
- All custom exceptions live in `src/utils/exceptions.py` and inherit from a common
  `MLPipelineError` base:
  `DataIngestionError`, `DataValidationError`, `FeatureEngineeringError`,
  `ModelTrainingError`, `ModelLoadError`, `PredictionError`.
- Never catch bare `Exception` except at the top-level API boundary (`api/exception_handlers.py`)
  and Airflow task boundary, where it's logged and re-raised or converted to an HTTP error.
- API translates exceptions to HTTP codes: validation → 422, model/load issues → 503,
  unexpected → 500 (logged with full traceback, generic message returned to caller).

### 3.6 Configuration
- Secrets and environment-specific values → `.env`, loaded via `src/config/settings.py`
  (`pydantic-settings.BaseSettings`). Never read `os.environ` directly outside this file.
- Hyperparameters, thresholds, file paths → `src/config/config.yaml`, loaded once and
  passed explicitly into functions (no hidden global state).

### 3.7 Testing
- `pytest`, minimum **80% coverage** on `src/` and `api/`, enforced in CI (`pytest --cov --cov-fail-under=80`).
- Unit tests use fixtures from `tests/conftest.py`; no test hits the network, a real database, or AWS — mock/stub those boundaries.
- One integration test suite spins up the FastAPI app with a fixture model and hits `/predict`.

### 3.8 Dependency management
- `requirements.txt` (runtime) and `requirements-dev.txt` (tooling) — every package **pinned**
  to an exact version (`==`), not a range. Regenerate with `pip-compile` if you add a package.

### 3.9 Object-oriented architecture & simplicity rules

Every pipeline stage is a **class**, not a loose function — but the OOP is kept deliberately
thin. The goal is readable, swappable components, not a framework.

**The one shared abstraction — `PipelineStage`** (`src/pipeline/base.py`):
```python
class PipelineStage(ABC):
    """Base class for every stage the Airflow DAG can call."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Execute this stage and return its output."""
```
Every stage class (`DataIngestor`, `DataValidator`, `DataCleaner`, `FeatureBuilder`,
`ModelTrainer`, `ModelEvaluator`, …) inherits from `PipelineStage` and implements `run()`.
This is the **only** reason inheritance exists in this project: it lets the Airflow DAG
call every stage the same way (`stage.run(data)`) without knowing its internals.

**Hard rules to keep this simple, not clever:**
- **One level of inheritance, maximum.** Concrete stages inherit directly from
  `PipelineStage`. Never subclass a subclass.
- **Composition over inheritance everywhere else.** A class that needs a config object or a
  logger takes it as a constructor argument — it doesn't inherit it from a mixin.
- **One class = one responsibility.** If a class's constructor takes more than ~3-4
  dependencies, or a method is doing two distinct jobs, split it.
- **No design patterns for their own sake.** No factories, singletons, builders, or
  registries unless there's a concrete need (e.g. swapping XGBoost for RandomForest is a
  legitimate reason for a small `ModelTrainer` strategy — a `PipelineStageFactory` that
  only ever builds one thing is not).
- **Methods stay short** — roughly 30 lines as a soft ceiling. A longer method is a sign it
  should call a private helper method, not that it needs a comment block.
- **Dependency injection via constructor**, never global state or module-level singletons
  (this also makes every class trivial to unit test — pass in a fake config/model).
- **No premature abstraction.** Don't build a plugin system, an event bus, or an interface
  for something this project only ever does one way. Add the abstraction when a second,
  real implementation shows up — not before.

---

## 4. Data Contracts (`src/schemas/models.py`)

These are the exact shapes every stage must produce/consume. Define them once as Pydantic
models and import everywhere — never redefine a schema inline in a pipeline step.

```python
class RawCustomerRecord(BaseModel):
    customer_id: str
    age: int
    tenure_months: int
    monthly_charges: float
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    # ... one field per raw column, typed, nothing inferred

class ProcessedCustomerRecord(RawCustomerRecord):
    # same fields, but guaranteed: no nulls, correct dtypes, ranges enforced
    pass

class FeatureVector(BaseModel):
    customer_id: str
    features: dict[str, float]   # final numeric feature set fed to the model

class PredictionRequest(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    contract_type: Literal["month-to-month", "one-year", "two-year"]

class PredictionResponse(BaseModel):
    customer_id: str | None
    prediction: Literal[0, 1]
    probability: float = Field(ge=0.0, le=1.0)
    model_version: str
```

**Rule:** `api/schemas.py` imports `PredictionRequest`/`PredictionResponse` from
`src/schemas/models.py` rather than redeclaring them — the API layer must never drift from
the pipeline's data contract.

---

## 5. Module Contracts

Each entry is the exact class an agent should implement — its base class, its `run()`
input/output, and its one responsibility. No class does more than one of these.

| File | Class | Inherits | `run()` signature | Responsibility |
|---|---|---|---|---|
| `data_ingestion/extract.py` | `DataIngestor` | `PipelineStage` | `(source_path: str) -> pd.DataFrame` | Read raw data only. No cleaning. |
| `validation/validate.py` | `DataValidator` | `PipelineStage` | `(df: pd.DataFrame) -> pd.DataFrame` | Run Great Expectations suite; raise `DataValidationError` on failure; return df unchanged on pass. |
| `data_processing/clean.py` | `DataCleaner` | `PipelineStage` | `(df: pd.DataFrame) -> pd.DataFrame` | Nulls, dtypes, dedup. No feature creation. |
| `data_processing/transform.py` | `DataTransformer` | `PipelineStage` | `(df: pd.DataFrame) -> pd.DataFrame` | Encoding, scaling. No feature *creation*. |
| `features/feature_engineering.py` | `FeatureBuilder` | `PipelineStage` | `(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]` | Return `(X, y)`. No I/O. |
| `training/train.py` | `ModelTrainer` | `PipelineStage` | `(X, y, params: dict) -> tuple[Any, dict]` | Fit model; return `(model, metrics)`; log both to MLflow inside `run()`. |
| `training/tune.py` | `HyperparameterTuner` | `PipelineStage` | `(X, y) -> dict` | Return best param dict only — does not train the final model. |
| `evaluation/evaluate.py` | `ModelEvaluator` | `PipelineStage` | `(model, X_test, y_test) -> bool` | Return `True` iff metrics clear the promotion threshold in `config.yaml`. |
| `api/model_loader.py` | `ModelLoader` | *(none — not an Airflow stage)* | `load_production_model() -> Any` | Fetch the `Production`-stage model from the MLflow Registry; cache in memory. |

Every class above: constructor takes `config: PipelineConfig` (see 3.9) plus any stage-specific
dependency, full type hints, Google docstring on the class and on `run()`, raises the matching
exception from `src/utils/exceptions.py` on failure, and has a corresponding unit test with
both a happy-path and a failure-path case. `ModelLoader` is the one exception to the
`PipelineStage` pattern — it's used by the API, not the DAG, so it doesn't need `run()`.

---

## 6. API Specification (`api/main.py`)

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` | Liveness probe, no model load. |
| GET | `/model/info` | — | `{"version": str, "trained_at": str, "metrics": dict}` | Reads MLflow Registry metadata. |
| POST | `/predict` | `PredictionRequest` | `PredictionResponse` | Single record. |
| POST | `/predict/batch` | `list[PredictionRequest]` | `list[PredictionResponse]` | Max 500 records/request (config). |

- Model is loaded **once** at startup via a FastAPI lifespan handler, not per-request.
- Every request is logged: request id, latency, prediction, model version (no PII beyond
  what's in the request itself).

---

## 7. Pipeline Orchestration (Airflow)

`airflow/dags/ml_pipeline_dag.py` — one task per module contract in Section 5, wired in the
exact order of Section 1's diagram. Each task:
- Instantiates its stage class and calls `stage.run(data)` — no business logic lives in the
  DAG file itself, it only wires stages together.
- Uses `PythonOperator` with `retries=1`, `retry_delay=5min`.
- The `ModelEvaluator` task branches (`BranchPythonOperator`): pass → `register_model` task
  (promotes to `Staging`); fail → `notify_failure` task, DAG ends without touching the
  registry.

---

## 8. MLflow Conventions

- Experiment name: `<project_name>/<model_type>` e.g. `churn-prediction/xgboost`.
- Run name: `<git_short_sha>-<timestamp>`.
- Always logged: all hyperparameters used, all metrics from `evaluate_model`, the
  `requirements.txt` hash, and the fitted model with its input signature.
- Registry stages used: `None → Staging → Production → Archived`. Promotion to `Production`
  is a **manual** step in this portfolio project (documented in README), not automatic —
  keeps a human in the loop, which is worth calling out in interviews.

---

## 9. CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

```
push/PR → lint (ruff, black --check, isort --check)
        → type-check (mypy --strict)
        → unit tests (pytest --cov, fail under 80%)
        → [only on main, only if above pass]
        → build Docker image
        → push to Amazon ECR
        → deploy to EC2 (pull + restart container)
```
Each stage is a separate job; later jobs `needs:` earlier ones. Secrets
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `EC2_HOST`, `EC2_SSH_KEY`) come from GitHub
Actions secrets, never committed.

---

## 10. AWS Deployment Contract

Required env vars on EC2 (documented in `.env.example`, values injected at deploy time —
never hardcoded, never committed):
`AWS_REGION`, `ECR_REPOSITORY_URI`, `MLFLOW_TRACKING_URI`, `MODEL_S3_BUCKET`, `ENVIRONMENT`.

IAM policy for the deploy role: least-privilege — `ecr:GetDownloadUrlForLayer`,
`ecr:BatchGetImage`, `s3:GetObject`/`PutObject` scoped to the one bucket, nothing broader.
Security group on EC2: inbound 443 (or 80 behind nginx) and 22 restricted to a known IP
range only — never `0.0.0.0/0` on 22.

---

## 11. Implementation Order for a Coding Agent

Work in this order; each phase should be fully tested (unit tests green, coverage ≥80%,
lint/type-check clean) before moving to the next:

1. **Foundations** — `src/config/settings.py`, `src/config/config.yaml`, `src/schemas/models.py`, `src/utils/logger.py`, `src/utils/exceptions.py`.
2. **Ingestion → Validation → Processing → Features** — Section 5 rows 1–5, each with tests.
3. **Training + Evaluation** — Section 5 rows 6–8, wired to a local MLflow tracking URI.
4. **Airflow DAG** — wire the phase-2/3 functions together per Section 7.
5. **API** — `api/`, Section 6, against a model fixture (mock MLflow load in tests).
6. **Docker** — Dockerfile + docker-compose for local API + MLflow.
7. **CI/CD** — `.github/workflows/ci-cd.yml` per Section 9.
8. **AWS** — ECR/EC2 per Section 10, deployment docs in `deployment/aws/`.

## 12. Definition of Done (per module)
- [ ] Full type hints, passes `mypy --strict`
- [ ] Google-style docstrings on the class and all public methods
- [ ] Pipeline stages inherit from `PipelineStage`; no extra inheritance levels added
- [ ] Constructor takes its dependencies as arguments — no global state, no singletons
- [ ] Uses shared logger, no `print()`
- [ ] Raises the correct custom exception on failure paths
- [ ] Unit test file exists at the mirrored `tests/unit/` path, happy + failure cases
- [ ] No hardcoded paths, secrets, or magic numbers (pull from `settings` or `config.yaml`)
- [ ] `black`, `isort`, `ruff` clean
