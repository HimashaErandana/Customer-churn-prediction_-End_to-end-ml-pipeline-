# MLOps Production Pipeline

End-to-end machine learning pipeline with MLOps practices and AWS deployment.

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the single source of truth on structure,
> naming, and build order. This project is built phase by phase, each phase fully
> tested before moving on.

## Current status

Phases 1–5 of the ARCHITECTURE.md implementation order are complete:
foundations (config/settings/schemas/logging/exceptions), the four data stages,
training + evaluation, the Airflow DAG (parquet handoff, registers passing models
to Staging), and the FastAPI service. Config-driven hyperparameters and
thresholds live in `src/config/config.yaml`.

Next up: Phase 6 (Docker), Phase 7 (CI/CD), Phase 8 (AWS).
