import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Evidently import paths differ across versions.
# Keep compatible with evidently==0.7.20 and older variants.
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except Exception:  # pragma: no cover
    from evidently import Report  # type: ignore
    from evidently.presets import DataDriftPreset  # type: ignore


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    ref = ref.dropna()
    cur = cur.dropna()

    if ref.empty or cur.empty:
        return 0.0

    # Quantile-based bins are more stable for skewed numeric features
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    # If near-constant feature, PSI isn't meaningful
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.clip(ref_dist, eps, 1.0)
    cur_dist = np.clip(cur_dist, eps, 1.0)

    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def compute_psi_by_col(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> Dict[str, float]:
    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(ref_df[c])]

    psi: Dict[str, float] = {}
    for c in numeric_cols:
        psi[c] = psi_numeric(ref_df[c], cur_df[c])
    return psi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference dataset CSV")
    p.add_argument("--current", required=True, help="Current dataset CSV or JSONL")
    p.add_argument("--out_dir", default="reports", help="Output directory")
    p.add_argument(
        "--psi_threshold",
        type=float,
        default=None,
        help="PSI threshold for drift gate (overrides env DRIFT_PSI_THRESHOLD).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(args.reference)

    if args.current.endswith(".jsonl"):
        cur_df = pd.read_json(args.current, lines=True)
    else:
        cur_df = pd.read_csv(args.current)

    # Evidently report (HTML)
    report = Report([DataDriftPreset(method="psi")])
    result = report.run(reference_data=ref_df, current_data=cur_df)
    html_path = out_dir / "drift_report.html"
    result.save_html(str(html_path))

    # PSI gate
    psi_by_col = compute_psi_by_col(ref_df, cur_df)

    if args.psi_threshold is not None:
        threshold = float(args.psi_threshold)
    else:
        threshold = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

    drifted = {k: v for k, v in psi_by_col.items() if v > threshold}
    max_feature = max(psi_by_col, key=psi_by_col.get) if psi_by_col else None

    summary = {
        "threshold": threshold,
        "max_psi": float(psi_by_col[max_feature]) if max_feature else 0.0,
        "max_psi_feature": max_feature,
        "n_drifted_features": int(len(drifted)),
        "drifted_features": drifted,
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_df)),
        "html_report": str(html_path),
    }

    json_path = out_dir / "drift_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    if drifted:
        print(
            f"Drift detected: {len(drifted)} features exceed PSI>{threshold}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
