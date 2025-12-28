# .github/scripts/check_accuracy_gate.py

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_METRICS_PATH = "artifacts/metrics/metrics.json"

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "cmapss_rul_xgb_optuna"

DEFAULT_METRIC_NAME = "r2"
DEFAULT_METRIC_MODE = "max"
DEFAULT_THRESHOLD = 0.90


def _mode_for_metric(metric_name: str) -> str:
    m = metric_name.lower().strip()
    if m in {"rmse", "mae"}:
        return "min"
    return "max"


def _fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


def _pass(msg: str) -> None:
    print(msg)
    sys.exit(0)


def read_metrics_json(metrics_path: Path) -> Dict[str, Any]:
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "metrics" not in obj or not isinstance(obj["metrics"], dict):
        raise ValueError("metrics.json missing top-level 'metrics' dict.")
    return obj


def best_metric_value_mlflow(runs, metric_name: str, mode: str) -> float:
    vals = []
    for r in runs:
        if metric_name in r.data.metrics:
            vals.append(float(r.data.metrics[metric_name]))
    if not vals:
        raise SystemExit(
            f"No runs with metric {metric_name!r} found. Ensure training logs it and experiment name is correct."
        )
    return min(vals) if mode == "min" else max(vals)


def main() -> None:
    metric_name = os.getenv("METRIC_NAME", DEFAULT_METRIC_NAME).strip()

    # Backward compatibility: accept old env names used in your workflow
    threshold_str = os.getenv("METRIC_THRESHOLD", os.getenv("ACCURACY_THRESHOLD", str(DEFAULT_THRESHOLD)))
    try:
        threshold = float(threshold_str)
    except ValueError:
        _fail(f"Invalid threshold value {threshold_str!r}. Must be float.")

    mode = os.getenv("METRIC_MODE", "").strip().lower() or _mode_for_metric(metric_name)
    if mode not in {"max", "min"}:
        _fail("METRIC_MODE must be 'max' or 'min'.")

    metrics_path = Path(os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH))

    # Preferred path: deterministic file-based gate
    if metrics_path.exists():
        obj = read_metrics_json(metrics_path)
        metrics = obj["metrics"]
        if metric_name not in metrics:
            _fail(f"metrics.json does not contain metric {metric_name!r}. Available: {list(metrics.keys())}")
        val = float(metrics[metric_name])
        print(f"Gate metric from {metrics_path}: {metric_name}={val:.6f} (mode={mode}) threshold={threshold:.6f}")

        if mode == "max" and val < threshold:
            _fail("Gate failed.")
        if mode == "min" and val > threshold:
            _fail("Gate failed.")
        _pass("Gate passed; deployment is allowed.")

    # Fallback: MLflow gate (legacy)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        _fail(f"Experiment {experiment_name!r} not found in MLflow store at {tracking_uri!r}, and metrics.json missing.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=2000,
    )

    best = best_metric_value_mlflow(runs, metric_name, mode)
    print(f"Gate metric from MLflow: best {metric_name} (mode={mode}) in {experiment_name!r}: {best:.6f}")

    if mode == "max" and best < threshold:
        _fail("Gate failed.")
    if mode == "min" and best > threshold:
        _fail("Gate failed.")
    _pass("Gate passed; deployment is allowed.")


if __name__ == "__main__":
    main()
