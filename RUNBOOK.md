# Operational Runbook

## Train & validate
- Full pipeline: `make pipeline`
- Train only: `make train`
- Gate metrics file: `artifacts/metrics/metrics.json`
- Model artifact: `models/best_model.joblib`

## Serve API
- Start: `make api`
- Health: `GET /health`
- Metrics: `GET /metrics`

## CI quality gate
- The CI gate reads `artifacts/metrics/metrics.json`.
- If the configured metric threshold is not met, the workflow fails and deployment does not proceed.

## Rollback (Kubernetes)
- Rollback is performed by redeploying the previous image tag and verifying rollout status.
