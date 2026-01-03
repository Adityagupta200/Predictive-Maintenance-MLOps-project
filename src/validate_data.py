# src/validate_data.py

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import load_config, get_logger, load_cmapss_txt, cmapss_column_names

LOGGER = get_logger("validate_data")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]
    stats: Dict[str, object]


def _validate_dataframe(df: pd.DataFrame, *, source: str) -> ValidationResult:
    errors: List[str] = []
    expected_cols = cmapss_column_names()

    stats: Dict[str, object] = {
        "source": source,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

    # Column/schema validation (strict, because load_cmapss_txt assigns known names)
    if list(df.columns) != expected_cols:
        errors.append(
            f"{source}: columns mismatch. expected={expected_cols} got={list(df.columns)}"
        )

    # Basic null checks
    null_counts = df.isna().sum().to_dict()
    stats["null_counts"] = {k: int(v) for k, v in null_counts.items()}
    if any(v > 0 for v in null_counts.values()):
        errors.append(f"{source}: missing values found (see null_counts).")

    # Type / numeric sanity
    # engine_id, cycle must be integer-like and positive
    for col in ["engine_id", "cycle"]:
        if col not in df.columns:
            errors.append(f"{source}: missing required column {col!r}.")
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"{source}: {col} is not numeric dtype={df[col].dtype}.")
            continue
        if (df[col] <= 0).any():
            errors.append(f"{source}: {col} must be > 0.")
        # integer-likeness check (safe for floats read as 1.0 etc.)
        if not np.all(np.equal(np.mod(df[col].to_numpy(), 1), 0)):
            errors.append(f"{source}: {col} contains non-integer values.")

    # Duplicate (engine_id, cycle) check
    if {"engine_id", "cycle"}.issubset(df.columns):
        dup_mask = df.duplicated(subset=["engine_id", "cycle"])
        dup_count = int(dup_mask.sum())
        stats["duplicate_engine_cycle_rows"] = dup_count
        if dup_count > 0:
            errors.append(f"{source}: found {dup_count} duplicate (engine_id, cycle) rows.")

    # Cycle monotonicity per engine (non-decreasing after sorting)
    if {"engine_id", "cycle"}.issubset(df.columns) and len(df) > 0:
        sorted_df = df.sort_values(["engine_id", "cycle"], kind="mergesort")
        bad = 0
        for eng_id, g in sorted_df.groupby("engine_id", sort=False):
            cycles = g["cycle"].to_numpy()
            if len(cycles) == 0:
                continue
            if cycles.min() != 1:
                errors.append(f"{source}: engine_id={int(eng_id)} does not start at cycle=1.")
            if np.any(np.diff(cycles) < 0):
                bad += 1
        stats["engines_with_decreasing_cycles"] = int(bad)
        if bad > 0:
            errors.append(f"{source}: {bad} engines have decreasing cycle values after sorting.")

    # Finite numeric check for sensors/settings
    numeric_cols = [c for c in df.columns if c not in {"engine_id", "cycle"}]
    if numeric_cols:
        arr = df[numeric_cols].to_numpy(dtype=float, copy=False)
        non_finite = int(np.size(arr) - np.isfinite(arr).sum())
        stats["non_finite_numeric_values"] = non_finite
        if non_finite > 0:
            errors.append(f"{source}: found {non_finite} non-finite numeric values (inf/nan).")

    return ValidationResult(ok=(len(errors) == 0), errors=errors, stats=stats)


def main(config_path: str) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    paths_cfg = config["paths"]

    data_dir = Path(data_cfg["raw_dir"])
    processed_dir = Path(paths_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    fd_sets: List[str] = list(data_cfg["fd_sets"])
    results: List[Dict[str, object]] = []
    ok = True
    all_errors: List[str] = []

    for fd in fd_sets:
        train_path = data_dir / f"train_{fd}.txt"
        df = load_cmapss_txt(train_path)
        res = _validate_dataframe(df, source=str(train_path))
        results.append({"ok": res.ok, "errors": res.errors, "stats": res.stats})
        if not res.ok:
            ok = False
            all_errors.extend(res.errors)

    report_path = processed_dir / "data_validation.json"
    report = {
        "ok": ok,
        "fd_sets": fd_sets,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote data validation report to %s", report_path)

    if not ok:
        for e in all_errors[:50]:
            LOGGER.error(e)
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    main(args.config)
