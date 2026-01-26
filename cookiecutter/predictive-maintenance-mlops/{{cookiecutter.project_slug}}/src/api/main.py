import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .model_loader import load_model
from .observability import (
    RequestContextMiddleware,
    audit_prediction_event,
    observe_inference_latency,
    setup_logging,
)
from .schemas import BatchPredictionResponse, PredictionRequest, PredictionResponse

setup_logging()

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
app.add_middleware(RequestContextMiddleware)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

model = load_model(MODEL_PATH)


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    # Prometheus scrape endpoint
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict_single(payload: PredictionRequest, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build DF exactly once so inference timing is stable
    df = pd.DataFrame([payload.features])

    start = time.perf_counter()
    try:
        y_pred = model.predict(df)
        pred_value = float(y_pred[0])

        infer_seconds = time.perf_counter() - start
        observe_inference_latency(model_version=MODEL_VERSION, outcome="success", seconds=infer_seconds)

        audit_prediction_event(
            request_id=getattr(request.state, "request_id", "unknown"),
            model_version=MODEL_VERSION,
            latency_ms=infer_seconds * 1000.0,
            inputs=payload.features,
            output={"prediction": pred_value},
        )
        return PredictionResponse(prediction=pred_value)

    except Exception as exc:
        infer_seconds = time.perf_counter() - start
        observe_inference_latency(model_version=MODEL_VERSION, outcome="error", seconds=infer_seconds)

        audit_prediction_event(
            request_id=getattr(request.state, "request_id", "unknown"),
            model_version=MODEL_VERSION,
            latency_ms=infer_seconds * 1000.0,
            inputs=payload.features,
            output={"error": str(exc)},
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payloads: List[PredictionRequest], request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([p.features for p in payloads])

    start = time.perf_counter()
    try:
        y_pred = model.predict(df)
        pred_list = [float(v) for v in y_pred]

        infer_seconds = time.perf_counter() - start
        observe_inference_latency(model_version=MODEL_VERSION, outcome="success", seconds=infer_seconds)

        audit_prediction_event(
            request_id=getattr(request.state, "request_id", "unknown"),
            model_version=MODEL_VERSION,
            latency_ms=infer_seconds * 1000.0,
            inputs={"batch_size": len(payloads)},
            output={"predictions_count": len(pred_list)},
        )
        return BatchPredictionResponse(predictions=pred_list)

    except Exception as exc:
        infer_seconds = time.perf_counter() - start
        observe_inference_latency(model_version=MODEL_VERSION, outcome="error", seconds=infer_seconds)

        audit_prediction_event(
            request_id=getattr(request.state, "request_id", "unknown"),
            model_version=MODEL_VERSION,
            latency_ms=infer_seconds * 1000.0,
            inputs={"batch_size": len(payloads)},
            output={"error": str(exc)},
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail=str(exc))
