# MLflow Experiments and Comparisons

This directory contains Jupyter notebooks for MLflow experiment tracking and run comparisons as per production requirements.

## 01_mlflow_experiments.ipynb

Execute after starting MLflow UI: `mlflow ui --host 127.0.0.1 --port 5000`.

- Run the notebook to log training runs.
![MLflow UI runs list](<screenshot of MLflow UI runs list.png>).
- Artifact: `notebooks/artifacts/runs_leaderboard.csv` – generated from notebook, shows top runs by metrics.

Proof command: `head -n 20 notebooks/artifacts/runs_leaderboard.csv`

## 02_compare_runs.ipynb

Loads runs from MLflow and generates comparison leaderboard.

- Run after 01_mlflow_experiments.ipynb.
- Compares params, metrics, artifacts across runs.
![leaderboard CSV preview](image.png)

![p95 latency graph](prometheus_promql_query_graph.png)
*p95 latency graph*

![evidently drift html report](drift-reports_html.png)
*Evidently AI HTML drift report*

All notebooks use `requirements-notebooks.txt` and integrate with DVC/MLflow for reproducibility.