# .github/scripts/check_accuracy_gate.py
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT = "tabular-classification"
DEFAULT_THRESHOLD = 0.90
DEFAULT_TRACKING_URI = "file:./mlruns"
METRIC_NAME = "test_accuracy"


def get_best_metric(experiment_name: str, metric_name: str) -> float:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment {experiment_name!r} not found in MLflow.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_name} IS NOT NULL",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No runs with metric {metric_name!r} found.")

    return float(runs[0].data.metrics[metric_name])


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    threshold = float(os.getenv("ACCURACY_THRESHOLD", DEFAULT_THRESHOLD))

    mlflow.set_tracking_uri(tracking_uri)
    best = get_best_metric(experiment_name, METRIC_NAME)
    print(f"Best {METRIC_NAME} in experiment {experiment_name!r}: {best:.4f}")

    if best < threshold:
        print(
            f"Accuracy gate failed: {METRIC_NAME}={best:.4f} "
            f"< threshold {threshold:.2f}"
        )
        sys.exit(1)

    print("Accuracy gate passed; deployment is allowed.")


if __name__ == "__main__":
    main()
