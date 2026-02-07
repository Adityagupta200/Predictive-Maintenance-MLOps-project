### 1. Pipeline & CI/CD
- [ ] **Data Reproducibility**: `dvc repro` generates identical `train.csv` / `val.csv` hashes.
- [ ] **Accuracy Gate**: Model validation confirms accuracy > 90%.
- [ ] **Rollback Capability**: Deployment supports automated rollback on failure.

### 2. Infrastructure & Security
- [ ] **K8s Resources**: `deployment.yaml` and `service.yaml` apply without error.
- [ ] **IRSA Setup**: `drift-dvc-reader` ServiceAccount created and annotated with AWS Role ARN (for S3 access).

### 3. Monitoring & Observability
- [ ] **Latency Target**: p95 API latency is < 200ms under load.
- [ ] **Drift Thresholds**: Drift check job completes successfully (PSI < 0.2).
- [ ] **Structured Logging**: Loguru outputs JSON with `request_id`, `model_version`, and `inputs`.

## Required Proof Artifacts
*Attach evidence as defined in `testing-production.pdf` & `Testing_fourth_point.pdf`*

- [ ] **Grafana**: Screenshot of p95 latency panel (< 200ms).
- [ ] **Prometheus**: Screenshot showing the API Target is "UP".
- [ ] **Drift Report**: Attached `drift_summary.json` or `data_drift.html` screenshot.
- [ ] **Audit Trail**: Snippet of JSON logs showing a successful prediction event.

## Standard Checklist
- [ ] Ran `make test` (Unit tests passed)
- [ ] Ran `make pipeline` (Integration verified)
- [ ] Ran `make docs` (Documentation updated)
- [ ] Ran pre-commit locally