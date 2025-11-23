# src/utils.py
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_config(config_path: str = "params.yaml") -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str) -> logging.Logger:
    """Simple logger with stdout handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers in DVC / notebooks

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def cmapss_column_names() -> List[str]:
    """
    Return standard C-MAPSS column names:
    id, cycle, 3 settings, 21 sensors. [web:6]
    """
    cols = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    return cols + sensor_cols


def load_cmapss_txt(path: Path) -> pd.DataFrame:
    """
    Load a C-MAPSS .txt file with whitespace separator and no header.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = cmapss_column_names()
    return df
