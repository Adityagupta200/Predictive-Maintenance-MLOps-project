# src/train_model.py
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

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

from src.utils import load_config, get_logger

LOGGER = get_logger("train_model")


def load_processed_data(processed_dir: Path) -> Dict[str, Any]:
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    with open(processed_dir / "meta_features.json", "r") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    target_col = meta["target_col"]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feature_cols": feature_cols,
        "target_col": target_col,
    }


def create_pipeline(trial: optuna.Trial, random_state: int) -> Pipeline:
    params = {
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


def objective(
    trial: optuna.Trial,
    data: Dict[str, Any],
    experiment_name: str,
    random_state: int,
) -> float:
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        # Experiment already set in train_with_optuna, but safe to call again
        mlflow.set_experiment(experiment_name)

        pipeline = create_pipeline(trial, random_state=random_state)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        y_pred = pipeline.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))

        mlflow.log_params(trial.params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("residuals_std", float(np.std(y_val - y_pred)))

        return rmse


def _resolve_mlflow_settings(config: Dict[str, Any]) -> tuple[str, str, str | None]:
    mlflow_cfg = config["mlflow"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", mlflow_cfg["experiment_name"])
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        mlflow_cfg.get("registered_model_name", ""),
    ) or None

    return tracking_uri, experiment_name, registered_model_name


def train_with_optuna(config: Dict[str, Any], processed_dir: Path) -> optuna.Study:
    data = load_processed_data(processed_dir)

    train_cfg = config["training"]
    tracking_uri, experiment_name, _ = _resolve_mlflow_settings(config)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    study = optuna.create_study(direction="minimize", study_name=train_cfg["study_name"])

    mlflow_cb = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="rmse",
    )

    LOGGER.info(
        "Starting Optuna study '%s' for %d trials",
        train_cfg["study_name"],
        train_cfg["n_trials"],
    )

    study.optimize(
        lambda trial: objective(
            trial,
            data=data,
            experiment_name=experiment_name,
            random_state=train_cfg["random_state"],
        ),
        n_trials=train_cfg["n_trials"],
        timeout=train_cfg.get("timeout"),
        callbacks=[mlflow_cb],
        show_progress_bar=False,
    )

    LOGGER.info("Best trial: %s", study.best_trial)
    return study


def log_best_model(study: optuna.Study, config: Dict[str, Any], processed_dir: Path) -> None:
    data = load_processed_data(processed_dir)

    train_cfg = config["training"]
    tracking_uri, experiment_name, registered_model_name = _resolve_mlflow_settings(config)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    best_params = study.best_trial.params

    def pipeline_from_params(params: Dict[str, Any]) -> Pipeline:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=train_cfg["random_state"],
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

    X_train = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_train = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    with mlflow.start_run(run_name="best_model"):
        pipeline = pipeline_from_params(best_params)

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        mlflow.log_params(best_params)
        mlflow.log_metric("train_time_full_sec", train_time)

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    # Persist pipeline locally for API deployment
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "best_model.joblib")

    LOGGER.info("Saved best model pipeline to %s", models_dir / "best_model.joblib")


def main(config_path: str) -> None:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])

    study = train_with_optuna(config, processed_dir)
    log_best_model(study, config, processed_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to params.yaml",
    )
    args = parser.parse_args()
    main(args.config)
