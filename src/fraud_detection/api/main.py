"""FastAPI scoring service.

Loads the model registered under the ``champion`` alias in MLflow at startup, pinned to the
concrete version the alias points to. The registered model is the full inference pipeline
(features + classifier), so the API computes features exactly as training did. Requests go to
the model as a list of dicts, which takes its NumPy fast path (~2 ms per row instead of ~150 ms
through pandas; same scores).

Run locally:  uv run uvicorn fraud_detection.api.main:app --reload
"""

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .schemas import Health, Liveness, Prediction, Transaction, TransactionBatch

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
prediction_logger = logging.getLogger("fraud_detection.api.predictions")

DEFAULT_MODEL_URI = "models:/fraud-detection@champion"
DEFAULT_THRESHOLD = 0.5
ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/@]+)$")
VERSION_URI = re.compile(r"^models:/[^/@]+/(?P<version>\d+)$")

PROBABILITY = Histogram(
    "fraud_prediction_probability",
    "Predicted fraud probabilities.",
    buckets=[i / 20 for i in range(1, 21)],
)
PREDICTIONS = Counter("fraud_predictions", "Scored transactions.", ["model_version", "is_fraud"])


@dataclass(frozen=True)
class LoadedModel:
    pyfunc: Any  # mlflow.pyfunc.PyFuncModel (or anything with .predict and .metadata)
    uri: str  # concrete URI that was loaded, e.g. models:/fraud-detection/7
    version: str

    @property
    def metadata(self) -> dict:
        meta = getattr(self.pyfunc, "metadata", None)
        return getattr(meta, "metadata", None) or {}


@dataclass
class ServiceState:
    model_uri: str  # as configured
    model: LoadedModel | None = None
    threshold: float = DEFAULT_THRESHOLD
    load_error: str | None = None


def load_model(uri: str) -> LoadedModel:
    """Load a pyfunc model, resolving ``models:/name@alias`` to a pinned version first."""
    import mlflow

    version = None
    if match := ALIAS_URI.match(uri):
        mv = mlflow.MlflowClient().get_model_version_by_alias(match["name"], match["alias"])
        version = str(mv.version)
        uri = f"models:/{match['name']}/{version}"
    elif match := VERSION_URI.match(uri):
        version = match["version"]
    model = mlflow.pyfunc.load_model(uri)
    if version is None:  # runs:/ or other URIs: fall back to the run id
        run_id = getattr(model.metadata, "run_id", None)
        version = f"run:{run_id}" if run_id else "unknown"
    return LoadedModel(pyfunc=model, uri=uri, version=version)


def resolve_threshold(metadata: dict) -> float:
    """FRAUD_THRESHOLD env > model metadata > 0.5."""
    if env := os.getenv("FRAUD_THRESHOLD"):
        return float(env)
    if metadata.get("threshold") is not None:
        return float(metadata["threshold"])
    return DEFAULT_THRESHOLD


def _warm_up(model: LoadedModel) -> None:
    """Score one row at startup, so the first request does not pay for building the fast path."""
    try:
        model.pyfunc.predict([{"TransactionID": 0, "TransactionDT": 86_400, "TransactionAmt": 1.0}])
    except Exception:
        logger.exception("Warm-up prediction failed (model stays loaded)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = ServiceState(model_uri=os.getenv("MODEL_URI", DEFAULT_MODEL_URI))
    try:
        state.model = load_model(state.model_uri)
        state.threshold = resolve_threshold(state.model.metadata)
        logger.info(
            "Loaded model %s (version %s, threshold %.4f)",
            state.model.uri,
            state.model.version,
            state.threshold,
        )
        _warm_up(state.model)
    except Exception as exc:  # the API still starts so /health can report the problem
        state.load_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Could not load model %s: %s", state.model_uri, exc)
    app.state.service = state
    yield


app = FastAPI(title="Fraud Detection API", version="0.2.0", lifespan=lifespan)

Instrumentator(excluded_handlers=["/metrics", "/health.*"]).instrument(app).expose(
    app, include_in_schema=False
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d %.1fms request_id=%s",
            request.method,
            request.url.path,
            status,
            elapsed_ms,
            request_id,
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


def _service(request: Request) -> ServiceState:
    return request.app.state.service


def _health(state: ServiceState, status: str) -> Health:
    model = state.model
    return Health(
        status=status,
        model_loaded=model is not None,
        model_uri=state.model_uri,
        resolved_model_uri=model.uri if model else None,
        model_version=model.version if model else None,
        model_type=model.metadata.get("model_type") if model else None,
        threshold=state.threshold if model else None,
        detail=state.load_error,
    )


@app.get("/health/live", response_model=Liveness)
def live() -> Liveness:
    return Liveness(status="ok")


@app.get("/health/ready", response_model=Health)
def ready(request: Request, response: Response) -> Health:
    state = _service(request)
    if state.model is None:
        response.status_code = 503
        return _health(state, "not_ready")
    return _health(state, "ready")


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Back-compat: readiness info, but always 200."""
    return _health(_service(request), "ok")


def _score(request: Request, txs: list[Transaction]) -> list[Prediction]:
    state = _service(request)
    model = state.model
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train and promote one first."
        )
    records = [{**tx.features, "TransactionID": tx.TransactionID} for tx in txs]
    try:
        probas = np.asarray(model.pyfunc.predict(records), dtype=float).ravel()
    except (ValueError, KeyError) as exc:
        logger.exception("Prediction rejected input")
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    if len(probas) != len(txs):
        logger.error("Model returned %d scores for %d rows", len(probas), len(txs))
        raise HTTPException(status_code=500, detail="Prediction failed.")

    request_id = getattr(request.state, "request_id", None)
    log_predictions = prediction_logger.isEnabledFor(logging.INFO)
    predictions = []
    for tx, proba in zip(txs, probas.tolist(), strict=True):
        is_fraud = proba >= state.threshold
        PROBABILITY.observe(proba)
        PREDICTIONS.labels(model.version, str(is_fraud).lower()).inc()
        if log_predictions:  # one JSON line per transaction, shipped for drift monitoring
            prediction_logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "TransactionID": tx.TransactionID,
                        "fraud_probability": proba,
                        "is_fraud": is_fraud,
                        "threshold": state.threshold,
                        "model_version": model.version,
                    }
                )
            )
        predictions.append(
            Prediction(
                TransactionID=tx.TransactionID,
                fraud_probability=proba,
                is_fraud=is_fraud,
                threshold=state.threshold,
                model_version=model.version,
                model_uri=model.uri,
            )
        )
    return predictions


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction, request: Request) -> Prediction:
    return _score(request, [tx])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(txs: TransactionBatch, request: Request) -> list[Prediction]:
    """Score up to 1000 transactions in one model call; results keep the request order."""
    return _score(request, txs)
