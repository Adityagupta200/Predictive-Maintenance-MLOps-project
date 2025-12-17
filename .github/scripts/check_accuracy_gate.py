# .github/scripts/check_accuracy_gate.py
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "cmapss_rul_xgb_optuna"

# Regression gate defaults (align with train_model.py metrics)
DEFAULT_METRIC_NAME = "r2"     # options: r2, rmse, mae
DEFAULT_METRIC_MODE = "max"    # r2 -> max, rmse/mae -> min
DEFAULT_THRESHOLD = 0.90       # for r2; for rmse pick a ceiling like 20.0


def best_metric_value(runs, metric_name: str, mode: str) -> float:
    vals = []
    for r in runs:
        if metric_name in r.data.metrics:
            vals.append(float(r.data.metrics[metric_name]))

    if not vals:
        raise SystemExit(
            f"No runs with metric {metric_name!r} found. "
            f"Ensure training logs it and experiment name is correct."
        )

    return min(vals) if mode == "min" else max(vals)


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)

    metric_name = os.getenv("METRIC_NAME", DEFAULT_METRIC_NAME).strip()
    mode = os.getenv("METRIC_MODE", DEFAULT_METRIC_MODE).strip().lower()
    if mode not in {"max", "min"}:
        raise SystemExit("METRIC_MODE must be 'max' or 'min'.")

    # keep backward compat with ACCURACY_THRESHOLD used in workflow
    threshold = float(os.getenv("ACCURACY_THRESHOLD", os.getenv("METRIC_THRESHOLD", DEFAULT_THRESHOLD)))

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment {experiment_name!r} not found in MLflow store at {tracking_uri!r}.")

    # No filter_string here (your earlier error); just fetch recent runs and pick best.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=2000,
    )

    best = best_metric_value(runs, metric_name, mode)
    print(f"Best {metric_name} (mode={mode}) in experiment {experiment_name!r}: {best:.6f}")

    if mode == "max" and best < threshold:
        print(f"Gate failed: {metric_name}={best:.6f} < threshold {threshold:.6f}")
        sys.exit(1)

    if mode == "min" and best > threshold:
        print(f"Gate failed: {metric_name}={best:.6f} > threshold {threshold:.6f}")
        sys.exit(1)

    print("Gate passed; deployment is allowed.")


if __name__ == "__main__":
    main()
