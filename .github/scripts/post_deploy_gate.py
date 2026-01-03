# .github/scripts/post_deploy_gate.py
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score


def p95(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 95))


def to_jsonable(v):
    # Avoid NaN/Inf (invalid JSON numbers)
    if v is None:
        return None
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return to_jsonable(obj)

    path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"))
    parser.add_argument("--val-csv", default=os.getenv("VAL_CSV"))
    parser.add_argument("--n-requests", type=int, default=int(os.getenv("N_REQUESTS", "50")))
    parser.add_argument("--p95-ms-max", type=float, default=float(os.getenv("P95_MS_MAX", "200")))
    parser.add_argument("--r2-min", type=float, default=float(os.getenv("POST_DEPLOY_R2_MIN", "0.90")))
    parser.add_argument("--out", default=os.getenv("POST_DEPLOY_OUT", "artifacts/post_deploy_gate.json"))
    args = parser.parse_args()

    out_path = Path(args.out)

    report = {
        "ok": False,
        "base_url": args.base_url,
        "checks": {
            "health_ok": False,
            "latency_ok": None,
            "r2_ok": None,
            "n_requests": 0,
            "p95_ms": None,
            "p95_ms_max": args.p95_ms_max,
            "r2": None,
            "r2_min": args.r2_min,
        },
        "error": None,
    }

    try:
        if not args.base_url:
            raise ValueError("Missing --base-url / BASE_URL")
        if not args.val_csv:
            raise ValueError("Missing --val-csv / VAL_CSV")

        base = args.base_url.rstrip("/")

        # 1) Smoke
        r = requests.get(f"{base}/health", timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"/health failed: {r.status_code} {r.text}")
        report["checks"]["health_ok"] = True

        # 2) Load validation data
        df = pd.read_csv(args.val_csv)
        if len(df) == 0:
            raise RuntimeError("val csv is empty")

        target_col = "RUL" if "RUL" in df.columns else None

        y_true = None
        if target_col:
            y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
            X = df.drop(columns=[target_col])
        else:
            X = df

        n = min(args.n_requests, len(X))
        report["checks"]["n_requests"] = int(n)
        X = X.head(n)

        # 3) Latency + predictions
        lat_ms = []
        preds = []

        for i in range(n):
            row = X.iloc[i].to_dict()
            payload = {"features": {k: to_jsonable(v) for k, v in row.items()}}

            start = time.perf_counter()
            resp = requests.post(f"{base}/predict", json=payload, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000.0
            lat_ms.append(elapsed)

            if resp.status_code != 200:
                raise RuntimeError(f"/predict failed: {resp.status_code} {resp.text}")

            body = resp.json()
            if "prediction" not in body:
                raise RuntimeError(f"/predict response missing 'prediction': {body}")

            preds.append(float(body["prediction"]))

        report["checks"]["p95_ms"] = p95(lat_ms)

        if report["checks"]["p95_ms"] > args.p95_ms_max:
            report["checks"]["latency_ok"] = False
            report["ok"] = False
            return 1
        report["checks"]["latency_ok"] = True

        # 4) Optional R2 gate
        if y_true is not None:
            y_true = y_true[:n]
            r2 = float(r2_score(y_true, np.asarray(preds, dtype=float)))
            report["checks"]["r2"] = r2
            if r2 < args.r2_min:
                report["checks"]["r2_ok"] = False
                report["ok"] = False
                return 1
            report["checks"]["r2_ok"] = True
        else:
            report["checks"]["r2_ok"] = None

        report["ok"] = True
        return 0

    except Exception as e:
        report["ok"] = False
        report["error"] = {"type": type(e).__name__, "message": str(e)}
        return 1

    finally:
        write_report(out_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
