```markdown
# Predictive Maintenance API with Real-time Monitoring MLOps Project

Comprehensive MLOps project implementing end-to-end ML pipeline for predictive maintenance using AWS, Kubernetes, MLflow, DVC, Evidently, Prometheus, and GitHub Actions having the following details - 

    1. *End-to-End ML Pipeline Automation*
		-  Data validation & preprocessing with Pandas/NumPy (feature engineering, missing value
		 handling)
		-  Automated model training using Scikit-learn/XGBoost with Optuna for hyperparameter
		 tuning
		-  MLflow for experiment tracking, model registry, and metadata logging (accuracy, training
		 time, hyperparameters)
		-  DVC for dataset versioning and reproducibility
	2. *Cloud-Based Deployment Infrastructure*
		-  AWS EC2 for model hosting with FastAPI REST endpoints
		-   Docker containerization of entire application (model, API, dependencies)
		-  Kubernetes orchestration for scaling prediction services
		-  Terraform for Infrastructure-as-Code IaC) provisioning
	3. *CI/CD Implementation*
		-  GitHub Actions for automated testing PyTest and deployment
		-  Model validation gates checking accuracy thresholds 90%) before production
		 deployment
		-  Automated rollback to previous model version if monitoring detects performance
		 degradation
	4. *Production Monitoring System*
		-  Evidently AI dashboards tracking data drift PSI  0.2 triggers alerts)
		-  Prometheus/Grafana for API latency monitoring 95th percentile < 200ms)
		-  Custom logging with Loguru capturing prediction inputs/outputs for audit trails
	5. *Collaboration & Documentation*
		 - Cookiecutter template enforcing project structure consistency
		-  Sphinx auto-generated documentation from code annotations
		-  Jupyter Notebooks with MLflow UI integration showing experimental history

## Project Structure

```

.
├── .dvc/
│   ├── cache/
│   ├── tmp/
│   └── .gitignore
├── .github/
│   ├── eks/
│   │   └── cluster.yaml
│   ├── scripts/
│   │   ├── check_accuracy_gate.py
│   │   └── post_deploy_gate.py
│   ├── workflows/
│   │   ├── ci-cd.yaml
│   │   ├── docs-and-quality.yaml
│   │   ├── drift-monitoring.yaml
│   │   ├── notebook-smoke.yml
│   │   ├── rollback.yml
│   │   └── template-smoke.yml
│   └── pull_request_template.md
├── actions-runner/
├── artifacts/
│   ├── metrics/
│   │   └── metrics.json
│   ├── processed/
│   │   ├── data_validation.json
│   │   ├── meta_features.json
│   │   ├── train.csv
│   │   └── val.csv
│   └── CMaps/
│       ├── Damage Propagation Modeling.pdf
│       ├── readme.txt
│       ├── RUL_FD001.txt
│       ├── RUL_FD002.txt
│       ├── RUL_FD003.txt
│       ├── RUL_FD004.txt
│       ├── test_FD001.txt
│       ├── test_FD002.txt
│       ├── test_FD003.txt
│       ├── test_FD004.txt
│       ├── train_FD001.txt
│       ├── train_FD002.txt
│       ├── train_FD003.txt
│       └── train_FD004.txt
├── cookiecutter/
│   └── predictive-maintenance-mlops/
│       ├── {{cookiecutter.project_slug}}/
│       │   ├── .dvc/
│       │   ├── .github/
│       │   ├── artifacts/
│       │   ├── docs/
│       │   ├── infra/
│       │   ├── models/
│       │   ├── notebooks/
│       │   ├── reports/
│       │   ├── reports_from_cluster/
│       │   ├── src/
│       │   ├── tests/
│       │   ├── .dockerignore
│       │   ├── .dvcignore
│       │   ├── .gitignore
│       │   ├── .pre-commit-config.yaml
│       │   ├── CODE_OF_CONDUCT.md
│       │   ├── CONTRIBUTING.md
│       │   ├── dvc.lock
│       │   ├── dvc.yaml
│       │   ├── Makefile
│       │   ├── params.yaml
│       │   ├── pm-drift-dvc-s3-read.json
│       │   ├── pyproject.toml
│       │   ├── README.md
│       │   ├── request.json
│       │   ├── requirements-api.txt
│       │   ├── requirements-dev.txt
│       │   ├── requirements-notebooks.txt
│       │   ├── requirements.txt
│       │   ├── RUNBOOK.md
│       │   └── servicemonitor-api.yaml
│       ├── docs/
│       │   ├── _build/
│       │   └── source/
│       │       ├── api.rst
│       │       ├── conf.py
│       │       ├── index.rst
│       │       ├── pipeline.rst
│       │       └── runbook.rst
│       ├── hooks/
│       │   └── post_gen_project.py
│       ├── infra/
│       │   ├── docker/
│       │   │   └── DockerFile
│       │   └── k8s/
│       │       ├── deployment.yaml
│       │       ├── drift-check-job.yaml
│       │       └── drift-dvc-sa.yaml
│       ├── cookiecutter.json
│       ├── requirements-api.txt
│       ├── requirements-dev.txt
│       ├── requirements-notebooks.txt
│       ├── requirements.txt
│       ├── RUNBOOK.md
│       └── servicemonitor-api.yaml
├── infra/
│   ├── k8s/
│   │   ├── p95-latency-alert.yaml
│   │   └── service.yaml
│   └── terraform/
│       └── .gitignore
├── mlruns/
├── models/
│   ├── .gitignore
│   └── best_model.joblib
├── notebooks/
│   ├── _executed/
│   ├── .ipynb_checkpoints/
│   ├── mlruns/
│   ├── notebooks_artifacts/
│   │   └── runs_leaderboard.csv
│   ├── 01_mlflow_experiments.ipynb
│   └── 02_compare_runs.ipynb
├── reports/
│   ├── drift_report.html
│   └── drift_summary.json
├── reports_from_cluster/
│   ├── drift_report.html
│   └── drift_summary.json
├── src/
│   ├── __pycache__/
│   ├── api/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model_loader.py
│   │   ├── observability.py
│   │   └── schemas.py
│   └── monitoring/
│       ├── __pycache__/
│       ├── __init__.py
│       ├── data_preprocessing.py
│       ├── drift_check.py
│       ├── train_model.py
│       ├── utils.py
│       └── validate_data.py
├── tests/
│   ├── __pycache__/
│   └── test_imports.py
├── .pytest_cache/
├── .venv/
├── .dockerignore
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── build.log
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── dvc.lock
├── dvc.yaml
├── Makefile
├── mlflow.db
├── params.yaml
├── pm-drift-dvc-s3-read.json
├── pyproject.toml
├── README.md
├── request.json
├── requirements-api.txt
├── requirements-dev.txt
├── requirements-notebooks.txt
├── requirements.txt
├── RUNBOOK.md
└── servicemonitor-api.yaml

```

## 1. End-to-End Pipeline Automation

### 1A. Data Validation & Preprocessing (Pandas/NumPy)

Generated `artifacts/processed/data_validation.json`, `train.csv`, `val.csv`.

![ls -lah artifacts/processed](image-6.png)
*Generated artifacts from training*

Commands: `ls -lah artifacts/processed`; `cat artifacts/processed/data_validation.json` 

### 1B. Automated Training (Scikit-learn/XGBoost, Optuna)

Trained model saved as `models/best_model.joblib`.

![ls -lah models](image-7.png)
*Trained models*

![model prediction](terraform_predict_output-1.png)
*Model predictions*

Commands: `ls -lah models`; `python -c "import joblib; m = joblib.load('models/best_model.joblib'); print(type(m))"` 

### 1C. MLflow Tracking (Metrics/Params/Artifacts)

Experiments tracked in MLflow UI; exported `notebooks/artifacts/runs_leaderboard.csv`.

**Attach screenshot of MLflow UI runs list here as proof.**
![MLFlow UI runs list](<screenshot of MLflow UI runs list-2.png>)
*MLFlow UI runs list*

Commands: `ls -lah notebooks/artifacts`; `head -n 20 notebooks/artifacts/runs_leaderboard.csv` 

### 1D. DVC Dataset Versioning & Reproducibility

Versioned with `dvc.yaml`, `dvc.lock`; supports S3 remote.

![dvc dag](<dvc dag output-1.png>)
*Output of "dvc dag"*

![dvc status --cloud](image-8.png)
*Output of "dvc status --cloud"*

Commands: `dvc dag`; `dvc status --cloud`; `dvc repro` 

## 2. Cloud Deployment Infrastructure

### 2A. Docker Container (API)

API runs with health/metrics endpoints.

![terraform get health](terraform_get_health-2.png)
*Health check of deployed production link*

![Deployed link predict endpoint check](terraform_predict_output-2.png)
*Deployed link predict endpoint check*

Commands: `docker build -t pm-api-local -f DockerFile .`; `docker run --rm -p 8000:8000 pm-api-local`; `curl -fsS http://127.0.0.1:8000/health`; `curl -fsS http://127.0.0.1:8000/metrics | head` 

### 2B. Kubernetes on EKS (Ephemeral Demo)

Deployed to EKS cluster; in-cluster smoke test passed.

![kubectl get pods -o wide](<kubectl get pods-2.png>)
*Running pod in EKS cluster*

![In-cluster smoke test](<kubectl -n $K8SNAMESPACE logs svc-smoke-1.png>)
*In-cluster smoke test*

![Deployment rollout](<kubectl rollout history after-1.png>)
*Deployment rollout*

Commands: `kubectl -n default get pods -o wide`; `kubectl -n default rollout status deployment/predictive-maintenance-api`; `kubectl -n default logs svc-smoke` 

### 2C. Terraform Provisioning

IaC applied successfully.

![Public IP for running API](terraform_output-1.png)
*Public IP for running API from ```python terraform output```*

Commands: `cd infra/terraform`; `terraform init`; `terraform plan -out=tfplan`; `terraform apply -auto-approve tfplan`; `terraform output` 

## 3. CI/CD Implementation

### 3A. CI Checks (PyTest)

All tests passed locally (mirrors GitHub Actions).

![Tests using PyTest](<pytest command.png>)
*Successful tests using PyTest*

Command: `pytest -q` 

### 3B. Post-Deploy Gate (p95/Accuracy Thresholds)

Gate passed: R² ≥ 0.90, p95 ≤ 200ms; `artifacts/postdeploygate.json`.

![postdeploygate.json](image-9.png)
*Post-deploy artifact*

Command: `cat artifacts/postdeploygate.json` 

### 3C. Automated Rollback (K8s Rollout Undo)

Rollback tested: v2 → v1, health verified.

![Before rollout](<kubectl rollout history before-1.png>)
*Before rollout* 

![After rollout](<kubectl rollout history after-2.png>)
*After rollout of second deployment*

Command: `kubectl -n default rollout history deployment/predictive-maintenance-api` 

## 4. Production Monitoring System

### 4A. Loguru Audit Trail

JSON logs captured prediction events (event=prediction, request_id, model_version, inputs/outputs).

![Running pods in "monitoring" namespace](<kubectl get pods -n monitoring.png>)
*Running pods in "monitoring" namespace*

![Running API in monitoring namespace](<kubectl get servicemonitor.png>)
*Running API in monitoring namespace*

Commands: Send 25 predictions via `curl`; `POD=$(kubectl -n default get pods -l app=predictive-maintenance-api -o jsonpath="{.items[^0].metadata.name}"); kubectl -n default logs $POD --tail=200` 

### 4B. Prometheus Scrape & Grafana p95

ServiceMonitor applied; target UP; p95 visualized.

![Running targets on Prometheus](prometheus_targets-1.png)
*Running targets on Prometheus*

![P95-latency-graph](prometheus_promql_query_graph-2.png)
*P95-latency-graph*

Commands: `kubectl -n monitoring get servicemonitor`; `kubectl -n monitoring get pods -o wide` 

### 4C. p95 Alert Rule (200ms Threshold)

Alert applied; tested to FIRING.

![P95-API-Alert](prometheus_alerts-2.png)
*P95-Latency API Alert*

Command: `kubectl apply -f infra/k8s/p95-latency-alert.yaml` 

### 4D. Drift Detection (Evidently PSI 0.2)

Job executed; `reports/drift_report.html`, `reports/drift_summary.json`; PSI > 0.2 triggers non-zero exit.

![drift-report-html](drift-reports_html-1.png)
*HTML Drift Report*

Commands: `kubectl -n default apply -f infra/k8s/drift-check-job.yaml`; `kubectl cp POD:app/reports reports/from_cluster`; `ls -lah reports reports/from_cluster`; `cat reports/drift_summary.json` 

Pre-requisite for drift job: Create IRSA ServiceAccount via commands in ServiceAccountCreation.md.

## 5. Collaboration & Documentation

### 5A. Cookiecutter Template

Template generates project; passes pytest.

![Cookiecutter generated repo](<cookiecutter geenrated repo-1.png>)
*Cookiecutter generated repo*

![Successful tests run using PyTest in generated repo](<pytest -q cookiecutter generated repo-2.png>)
*Successful tests run using PyTest in generated repo*

Commands: `cookiecutter cookiecutter-project-slug/ -o tmp`; `cd tmp/...`; `pytest -q` 

### 5B. Sphinx Auto-Generated Docs

Docs built from code annotations.

![Sphinx Docs index.html](sphinx-docs-index-html-1.png)
*Sphinx Docs*

Command: `make -C docs html` 

### 5C. Notebooks with MLflow UI

Notebooks log to MLflow; runs visible in UI; `notebooks/artifacts/runs_leaderboard.csv`.

![MLFlow UI runs list](<screenshot of MLflow UI runs list-3.png>)
*MLFlow UI runs list*

Commands: `mlflow ui`; `jupyter lab` 

## Teardown (Ephemeral Resources)

- `eksctl delete cluster --name $CLUSTERNAME --region $AWSREGION`
- `cd infra/terraform && terraform destroy -auto-approve`
- `helm uninstall kube-prometheus-stack -n monitoring`
- `kubectl delete namespace monitoring`
- Delete K8s manifests: service.yaml, deployment.yaml, drift-check-job.yaml 

## Tools Stack

- ML: Scikit-learn, XGBoost, Optuna, MLflow
- Data: Pandas, NumPy, DVC
- Infra: AWS (EKS/EC2/S3), Docker, Kubernetes, Terraform
- CI/CD: GitHub Actions, PyTest
- Monitoring: Evidently (PSI 0.2), Prometheus/Grafana (p95 200ms), Loguru 
```