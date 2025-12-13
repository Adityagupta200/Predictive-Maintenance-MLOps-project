from pathlib import Path
from typing import Optional, Any
import joblib

def load_model(model_path: Path) -> Optional[Any]:
    if not model_path.exists():
        print(f"Warning: model file not found at {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        print(f"Error loading model from {model_path}: {exc}")
    return None