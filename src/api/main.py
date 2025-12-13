from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
from pathlib import Path

from .model_loader import load_model
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
)

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

MODEL_PATH = Path("models/best_model.joblib")
model = load_model(MODEL_PATH)


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(payload: PredictionRequest):
    df = pd.DataFrame([payload.features])
    try:
        y_pred = model.predict(df)
        return PredictionResponse(prediction=float(y_pred[0]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payloads: List[PredictionRequest]):
    df = pd.DataFrame([p.features for p in payloads])
    try:
        y_pred = model.predict(df)
        return BatchPredictionResponse(
            predictions=[float(v) for v in y_pred]
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
