from pathlib import Path
from typing import Any, Optional

from loguru import logger
import joblib


def load_model(model_path: Path) -> Optional[Any]:
    if not model_path.exists():
        logger.warning("Model file not found at {}", str(model_path))
        return None

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from {}", str(model_path))
        return model
    except Exception as exc:
        logger.exception("Error loading model from {}: {}", str(model_path), exc)
        return None
