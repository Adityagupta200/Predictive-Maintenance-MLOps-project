# src/data_preprocessing.py
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config, get_logger, load_cmapss_txt


LOGGER = get_logger("data_preprocessing")


def add_rul_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) per row:
    RUL = max_cycle_for_engine - current_cycle.
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature engineering:
      - cycle_norm: cycle / max_cycle_per_engine
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycle.replace(0, np.nan)
    df["cycle_norm"] = df["cycle_norm"].fillna(0.0)
    return df


def load_fd_dataset(
    data_dir: Path, fd_set: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load train/test/RUL files for a particular FD set (e.g., 'FD001').
    """
    train_path = data_dir / f"train_{fd_set}.txt"
    test_path = data_dir / f"test_{fd_set}.txt"
    rul_path = data_dir / f"RUL_{fd_set}.txt"

    LOGGER.info(f"Loading {fd_set} from {data_dir}")
    train_df = load_cmapss_txt(train_path)
    test_df = load_cmapss_txt(test_path)

    # RUL file has one value per engine in test set: RUL at last cycle. [web:2]
    rul_series = pd.read_csv(
        rul_path, sep=r"\s+", header=None
    ).iloc[:, 0].rename("RUL")

    return train_df, test_df, rul_series


def build_training_dataframe(
    data_dir: Path, fd_sets
) -> pd.DataFrame:
    """
    Concatenate training data from all requested FD sets,
    compute RUL and add basic features.
    """
    all_dfs = []
    for fd in fd_sets:
        train_df, _, _ = load_fd_dataset(data_dir, fd)
        train_df = add_rul_column(train_df)
        train_df = add_basic_features(train_df)
        train_df["fd_set"] = fd
        all_dfs.append(train_df)

    full_train = pd.concat(all_dfs, ignore_index=True)
    LOGGER.info(f"Combined train shape: {full_train.shape}")
    return full_train


def split_by_engine(
    df: pd.DataFrame,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/validation by engine_id to avoid leakage between sets.
    """
    engines = df["engine_id"].unique()
    train_eng, val_eng = train_test_split(
        engines, test_size=val_size, random_state=random_state
    )
    train_df = df[df["engine_id"].isin(train_eng)].reset_index(drop=True)
    val_df = df[df["engine_id"].isin(val_eng)].reset_index(drop=True)
    LOGGER.info(
        "Train engines: %d, Val engines: %d", len(train_eng), len(val_eng)
    )
    return train_df, val_df


def main(config_path: str) -> None:
    config = load_config(config_path)

    data_cfg = config["data"]
    paths_cfg = config["paths"]

    data_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(paths_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    fd_sets = data_cfg["fd_sets"]
    val_size = data_cfg["val_size"]
    random_state = data_cfg["random_state"]

    full_train = build_training_dataframe(data_dir, fd_sets)

    # Select feature and target columns
    target_col = data_cfg["target_col"]
    drop_cols = ["RUL", "engine_id", "fd_set"]
    feature_cols = [c for c in full_train.columns if c not in drop_cols]

    train_df, val_df = split_by_engine(
        full_train, val_size=val_size, random_state=random_state
    )

    LOGGER.info(
        "Final train shape: %s, val shape: %s",
        train_df.shape,
        val_df.shape,
    )

    # Save as CSV for DVC-tracked pipeline
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    meta_path = processed_dir / "meta_features.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Save minimal metadata (features + target)
    import json

    meta = {"feature_cols": feature_cols, "target_col": target_col}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("Saved processed data to %s", processed_dir)


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
