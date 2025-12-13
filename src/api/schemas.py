from pydantic import BaseModel
from typing import Dict, List

class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: float  # single numeric output

class BatchPredictionResponse(BaseModel):
    predictions: List[float]  # list of numeric outputs
