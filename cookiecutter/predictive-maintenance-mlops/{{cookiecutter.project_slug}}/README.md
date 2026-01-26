# Predictive Maintenance (CMAPSS) — MLOps Project

End-to-end MLOps pipeline for Remaining Useful Life (RUL) style predictive maintenance:
DVC pipeline (validate → preprocess → train), MLflow + Optuna for experiments, FastAPI for inference,
CI quality gate, and Kubernetes deployment.

## Repo quickstart

### 1) Setup
make setup

### 2) Reproduce training pipeline
make pipeline

Key outputs:
- artifacts/processed/train.csv, artifacts/processed/val.csv
- artifacts/metrics/metrics.json
- models/best_model.joblib

### 3) Run API locally
make api

Endpoints:
- GET /health
- GET /metrics
- POST /predict
- POST /predict/batch

## CI/CD
- PR/push triggers run tests + DVC pipeline + metrics gate.
- On main success, Docker image is built/pushed and deployed to Kubernetes (EKS).

## Docs
Build docs locally:
make docs
make docs-serve

## Contributing
See CONTRIBUTING.md and the PR template.
