import os
import time
import uuid
from typing import Any, Dict, Optional

from loguru import logger
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

SERVICE_NAME = os.getenv("SERVICE_NAME", "predictive-maintenance-api")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["service", "method", "path", "status_code"],
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["service", "method", "path", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 2, 5),
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions served",
    ["service", "model_version"],
)


def setup_logging() -> None:
    """
    Production-friendly structured JSON logs to stdout.
    """
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        serialize=True,
        level=os.getenv("LOG_LEVEL", "INFO"),
        backtrace=False,
        diagnose=False,
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Adds X-Request-ID and emits request metrics.
    """
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        dur = time.perf_counter() - start

        path = request.url.path
        method = request.method
        status = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(SERVICE_NAME, method, path, status).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(SERVICE_NAME, method, path, status).observe(dur)

        response.headers[self.header_name] = request_id
        return response


def audit_prediction_event(
    *,
    request_id: str,
    model_version: str,
    latency_ms: float,
    inputs: Dict[str, Any],
    output: Dict[str, Any],
) -> None:
    logger.bind(
        event="prediction",
        service=SERVICE_NAME,
        request_id=request_id,
        model_version=model_version,
        latency_ms=round(latency_ms, 3),
        inputs=inputs,
        output=output,
    ).info("prediction_served")

    MODEL_PREDICTIONS_TOTAL.labels(SERVICE_NAME, model_version).inc()
