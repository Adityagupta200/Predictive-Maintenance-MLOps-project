# CONTRIBUTING.md

## Project Overview

This project implements Predictive Maintenance API with real-time monitoring at production level having the following details -

1. **End-to-End ML Pipeline Automation**
	-  Data validation & preprocessing with Pandas/NumPy (feature engineering, missing value handling)
	- Automated model training using Scikit-learn/XGBoost with Optuna for hyperparameter tuning
	-  MLflow for experiment tracking, model registry, and metadata logging (accuracy, training time, hyperparameters)
	- DVC for dataset versioning and reproducibility
2. **Cloud-Based Deployment Infrastructure**
	-  AWS EC2 for model hosting with FastAPI REST endpoints
	-   Docker containerization of entire application (model, API, dependencies)
	-  Kubernetes orchestration for scaling prediction services
	-  Terraform for Infrastructure-as-Code IaC) provisioning
3. **CI/CD Implementation**
	-  GitHub Actions for automated testing PyTest and deployment
	-  Model validation gates checking accuracy thresholds 90%) before production deployment
	-  Automated rollback to previous model version if monitoring detects performance degradation
4. **Production Monitoring System**
	-  Evidently AI dashboards tracking data drift PSI  0.2 triggers alerts)
	-  Prometheus/Grafana for API latency monitoring 95th percentile < 200ms)
	-  Custom logging with Loguru capturing prediction inputs/outputs for audit trails
5. **Collaboration & Documentation**
	- Cookiecutter template enforcing project structure consistency
	-  Sphinx auto-generated documentation from code annotations
	- Jupyter Notebooks with MLflow UI integration showing experimental history

## Development Setup

- Create virtual environment: `python -m venv .venv`; activate with `source .venv/bin/activate` (Linux/Mac) or `.\\.venv\\Scripts\\Activate.ps1` (Windows).
- Install dependencies: `python -m pip install -U pip`, `python -m pip install -r requirements.txt`, `python -m pip install -r requirements-dev.txt`.
- Run local pipeline: `python -m src.monitoring.validate_data`, `python -m src.monitoring.data_preprocessing`, `python -m src.monitoring.train_model`.


## Testing Production Features

Test using ephemeral EKS clusters for production-like validation.

### 1. End-to-End Pipeline

![ls -lah models](./pngs/image-3.png)
*Generated model directory*

![runs_leaderboard.csv](./pngs/image-4.png)
*Artifact generated after run of "02_compare_runs.ipynb"*

![dvc dag](./pngs/dvc_dag_output.png)

*Output of "dvc dag"*

![dvc status --cloud](./pngs/image-5.png)

*Output of "dvc status --cloud"*

### 2. Cloud Deployment

![kubectl get pods -o wide](./pngs/kubectl_get_pods-1.png)
*Running pod in AWS EKS cluster*

![in-cluster smoke test logs](./pngs/kubectl_n_$K8SNAMESPACE_logs_svc-smoke.png)
*In-cluster smoke test logs*

![Docker health](./pngs/terraform_get_health-1.png)
*Health check of running API with infrastructure provisioned using Terraform*

![alt text](./pngs/terraform_predict_output.png)
*Predictions from running API*

![terraform_output.png](./pngs/terraform_output.png)
*IP of publically accessible API*

### 3. CI/CD Gates \& Rollback

![Pytest output](./pngs/pytest_q_cookiecutter_generated_repo-1.png)
*Successfull tests using Pytest*

![postdeploygate.json](./pngs/image-2.png)
*Post-Deploy artifact*

*Before rollout*
![Before rollout](./pngs/kubectl_rollout_history_before.png)

*After rollout*
![After rollout](./pngs/kubectl_rollout_history_after.png)

### 4. Production Monitoring

**Before `kubectl -n default apply -f infra/k8s/drift-check-job.yaml`:** Create IRSA ServiceAccount per ServiceAccountCreation.md (eksctl create iamserviceaccount for drift-dvc-reader).

![Prometheus Targets UP](./pngs/prometheus_targets.png)
*Prometheus Targets UP*

![Grafana p95 panel](./pngs/prometheus_promql_query_graph-1.png)
*Grafana p95 panel*

![alert FIRING at http://127.0.0.1:9090/alerts](./pngs/prometheus_alerts-1.png)
*alert FIRING at http://127.0.0.1:9090/alerts*

![reports_from_cluster/drift_report.html](./pngs/drift-reports_html.png)
*In-cluster generated Evidently AI drift report*

![drift-summary.json](./pngs/image-1.png)
*Drift summary*

### 5. Collaboration Tools

![cookiecutter generated repo](./pngs/cookiecutter_genrated_repo.png)
*Cookiecutter generated repo*

![Tests in generated repo](./pngs/pytest_q_cookiecutter_generated_repo.png)
*Tests in generated repo*

![docs-and-quality-workflow-run](./pngs/docs-ans-quality-ci-workflow.png)
*docs-and-quality-workflow-run*

![Sphinx docs](./pngs/sphinx-docs-index-html.png)
![Sphinx docs](./pngs/sphinx-docs-index-html2.png)
*Sphinx docs*

![MLFlow UI runs list](./pngs/screenshot_of_MLflow_UI_runs_list_1.png)
*MLFlow UI runs list*

## Pull Requests

- Run `pre-commit` hooks via `.pre-commit-config.yaml`.
- Update proofs in README.md with new artifacts.
- Target main branch; include production test screenshots.


## Cleanup Commands

- EKS: `eksctl delete cluster --name $CLUSTERNAME --region $AWSREGION`.
- Terraform: `cd infra/terraform; terraform destroy -auto-approve`.
- K8s/Monitoring: Uninstall kube-prometheus-stack, delete manifests/namespace.