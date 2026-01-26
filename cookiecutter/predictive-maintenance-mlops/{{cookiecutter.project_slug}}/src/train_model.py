import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import urlopen

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration import MLflowCallback
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, load_config

LOGGER = get_logger("train_model")


def load_processed_data(processed_dir: Path) -> Dict[str, Any]:
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    meta_path = processed_dir / "meta_features.json"
    if not meta_path.exists():
        meta_path = processed_dir / "metafeatures.json"

    if not meta_path.exists():
        raise SystemExit(
            f"Missing feature metadata. Expected {processed_dir/'meta_features.json'} "
            f"or {processed_dir/'metafeatures.json'}."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols") or meta.get("featurecols")
    target_col = meta.get("target_col") or meta.get("targetcol")

    if not feature_cols or not target_col:
        raise SystemExit(
            "Meta features missing required keys. Need feature_cols/target_col "
            "or featurecols/targetcol."
        )

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise SystemExit(f"train.csv missing required feature columns: {missing}")

    if target_col not in train_df.columns or target_col not in val_df.columns:
        raise SystemExit(f"Target column {target_col!r} not found in train/val CSVs.")

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_col].to_numpy()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


def build_pipeline(params: Dict[str, Any], random_state: int) -> Pipeline:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    residuals_std = float(np.std(y_true - y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "residuals_std": residuals_std}


def _mlflow_http_reachable(tracking_uri: str, timeout_sec: float = 2.0) -> bool:
    if not tracking_uri.startswith(("http://", "https://")):
        return True
    url = tracking_uri.rstrip("/") + "/api/2.0/mlflow/experiments/list"
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            return int(resp.status) < 500
    except Exception:
        return False


def _normalize_tracking_uri(tracking_uri: str) -> str:
    strict = os.getenv("MLFLOW_STRICT", "0").strip().lower() in {"1", "true", "yes"}
    if tracking_uri.startswith(("http://", "https://")) and not _mlflow_http_reachable(tracking_uri):
        if strict:
            raise SystemExit(
                f"MLflow tracking server unreachable at {tracking_uri}. "
                "Start MLflow server or set MLFLOW_TRACKING_URI=file:./mlruns."
            )
        fallback = "file:./mlruns"
        LOGGER.warning(
            "MLflow tracking server unreachable at %s. Falling back to local store %s",
            tracking_uri,
            fallback,
        )
        return fallback
    return tracking_uri


def _resolve_mlflow_settings(config: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    mlflow_cfg = config["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_cfg["experiment_name"])
    registered_model_name = (
        os.getenv("MLFLOW_REGISTERED_MODEL_NAME", mlflow_cfg.get("registered_model_name", "")) or None
    )
    tracking_uri = _normalize_tracking_uri(tracking_uri)
    return tracking_uri, experiment_name, registered_model_name


def train_with_optuna(config: Dict[str, Any], processed_dir: Path) -> optuna.Study:
    data = load_processed_data(processed_dir)
    train_cfg = config["training"]

    tracking_uri, experiment_name, _ = _resolve_mlflow_settings(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        pipeline = build_pipeline(params, random_state=int(train_cfg["random_state"]))

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            start = time.perf_counter()
            pipeline.fit(data["X_train"], data["y_train"])
            train_time = time.perf_counter() - start

            y_pred = pipeline.predict(data["X_val"])
            metrics = evaluate_regression(data["y_val"], y_pred)

            mlflow.log_params(params)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("train_time_sec", float(train_time))

            # Optimize RMSE (lower is better)
            return float(metrics["rmse"])

    study = optuna.create_study(
        direction="minimize",
        study_name=train_cfg["study_name"],
    )

    mlflow_cb = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="rmse",
    )

    LOGGER.info(
        "Starting Optuna study '%s' for %d trials",
        train_cfg["study_name"],
        int(train_cfg["n_trials"]),
    )

    study.optimize(
        objective,
        n_trials=int(train_cfg["n_trials"]),
        timeout=train_cfg.get("timeout"),
        callbacks=[mlflow_cb],
        show_progress_bar=False,
    )

    LOGGER.info("Best trial: %s", study.best_trial)
    return study


def write_metrics_json(*, metrics_path: Path, metric_payload: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metrics to %s", metrics_path)


def log_best_model_and_artifacts(study: optuna.Study, config: Dict[str, Any], processed_dir: Path) -> None:
    data = load_processed_data(processed_dir)
    train_cfg = config["training"]

    tracking_uri, experiment_name, registered_model_name = _resolve_mlflow_settings(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_params = dict(study.best_trial.params)
    random_state = int(train_cfg["random_state"])

    # 1) Deterministic gate metrics: fit on train split only, evaluate on val split.
    gate_pipeline = build_pipeline(best_params, random_state=random_state)

    start = time.perf_counter()
    gate_pipeline.fit(data["X_train"], data["y_train"])
    train_time = time.perf_counter() - start

    y_pred_val = gate_pipeline.predict(data["X_val"])
    gate_metrics = evaluate_regression(data["y_val"], y_pred_val)

    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "artifacts/metrics"))
    metrics_path = metrics_dir / "metrics.json"

    metric_payload = {
        "problem_type": "regression",
        "metric_mode_default": {"r2": "max", "rmse": "min", "mae": "min"},
        "metrics": gate_metrics,
        "train_time_sec": float(train_time),
        "best_params": best_params,
        "n_trials": int(train_cfg["n_trials"]),
        "study_name": train_cfg["study_name"],
    }
    write_metrics_json(metrics_path=metrics_path, metric_payload=metric_payload)

    # 2) Train final model on train+val for deployment artifact.
    X_full = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_full = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    final_pipeline = build_pipeline(best_params, random_state=random_state)

    with mlflow.start_run(run_name="best_model"):
        start_full = time.perf_counter()
        final_pipeline.fit(X_full, y_full)
        train_time_full = time.perf_counter() - start_full

        mlflow.log_params(best_params)
        mlflow.log_metric("train_time_full_sec", float(train_time_full))
        for k, v in gate_metrics.items():
            mlflow.log_metric(f"val_{k}", float(v))

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Canonical path used by API + CI
    joblib.dump(final_pipeline, models_dir / "best_model.joblib")
    # Backward-compatible path (older references)
    joblib.dump(final_pipeline, models_dir / "bestmodel.joblib")

    LOGGER.info("Saved best model pipeline to %s", models_dir / "best_model.joblib")


def main(config_path: str) -> None:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])

    study = train_with_optuna(config, processed_dir)
    log_best_model_and_artifacts(study, config, processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    main(args.config)
