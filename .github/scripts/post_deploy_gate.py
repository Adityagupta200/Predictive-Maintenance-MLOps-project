import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score


def _p95(values):
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 95))


def _to_jsonable(v):
    # pandas/numpy scalars -> python scalars
    if isinstance(v, (np.generic,)):
        return v.item()
    # NaNs are not valid JSON numbers
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:18000"))
    parser.add_argument("--val-csv", default=os.getenv("VAL_CSV", "artifacts/processed/val.csv"))
    parser.add_argument("--n-requests", type=int, default=int(os.getenv("N_REQUESTS", "50")))
    parser.add_argument("--p95-ms-max", type=float, default=float(os.getenv("P95_MS_MAX", "200")))
    parser.add_argument("--r2-min", type=float, default=float(os.getenv("POST_DEPLOY_R2_MIN", "0.90")))
    parser.add_argument("--out", default=os.getenv("POST_DEPLOY_OUT", "artifacts/post_deploy_gate.json"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Smoke check
    r = requests.get(f"{base}/health", timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"/health failed: {r.status_code} {r.text}")

    # 2) Load val.csv and build payloads
    val_path = Path(args.val_csv)
    if not val_path.exists():
        raise SystemExit(f"Missing {val_path}. Provide artifacts/processed/val.csv to run post-deploy checks.")

    df = pd.read_csv(val_path)

    y_true = None
    if "RUL" in df.columns:
        y_true = df["RUL"].astype(float).to_numpy()
        X = df.drop(columns=["RUL"])
    else:
        X = df

    if len(X) == 0:
        raise SystemExit("val.csv is empty.")

    n = min(args.n_requests, len(X))
    X = X.head(n)

    latencies_ms = []
    preds = []

    for i in range(n):
        row = X.iloc[i].to_dict()
        features = {k: _to_jsonable(v) for k, v in row.items()}
        payload = {"features": features}

        start = time.perf_counter()
        resp = requests.post(f"{base}/predict", json=payload, timeout=30)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if resp.status_code != 200:
            raise SystemExit(f"/predict failed: {resp.status_code} {resp.text}")

        body = resp.json()
        if "prediction" not in body:
            raise SystemExit(f"/predict response missing 'prediction': {body}")
        preds.append(float(body["prediction"]))

    # 3) Latency gate
    p95_ms = _p95(latencies_ms)
    checks = {"health_ok": True, "p95_ms": p95_ms, "p95_ms_max": args.p95_ms_max}

    if p95_ms > args.p95_ms_max:
        checks["latency_ok"] = False
        out_path.write_text(json.dumps({"ok": False, "checks": checks}, indent=2), encoding="utf-8")
        raise SystemExit(f"Latency gate failed: p95_ms={p95_ms:.2f} > {args.p95_ms_max:.2f}")
    checks["latency_ok"] = True

    # 4) Optional R2 gate
    if y_true is not None:
        y_true_n = y_true[:n]
        r2 = float(r2_score(y_true_n, np.asarray(preds, dtype=float)))
        checks["r2"] = r2
        checks["r2_min"] = args.r2_min
        if r2 < args.r2_min:
            checks["r2_ok"] = False
            out_path.write_text(json.dumps({"ok": False, "checks": checks}, indent=2), encoding="utf-8")
            raise SystemExit(f"R2 gate failed: r2={r2:.4f} < {args.r2_min:.4f}")
        checks["r2_ok"] = True
    else:
        checks["r2_skipped"] = True

    out_path.write_text(json.dumps({"ok": True, "checks": checks}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
