"""FastAPI scoring service.

Loads the model registered under the ``champion`` alias in MLflow at startup. The
registered model is the full Kedro inference pipeline (features + LightGBM), so the API
computes features exactly as training did.

Run locally:  uv run uvicorn fraud_detection.api.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import Health, Prediction, Transaction

logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/fraud-detection@champion")
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    try:
        import mlflow.pyfunc

        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        logger.info("Loaded model %s", MODEL_URI)
    except Exception as exc:  # the API still starts so /health can report the problem
        logger.warning("Could not load model %s: %s", MODEL_URI, exc)
    yield


app = FastAPI(title="Fraud Detection API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model_loaded=app.state.model is not None, model_uri=MODEL_URI)


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction) -> Prediction:
    if app.state.model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train and promote one first."
        )
    frame = pd.DataFrame([{"TransactionID": tx.TransactionID, **tx.features}])
    proba = float(pd.Series(app.state.model.predict(frame)).iloc[0])
    return Prediction(
        TransactionID=tx.TransactionID,
        fraud_probability=proba,
        is_fraud=proba >= THRESHOLD,
        threshold=THRESHOLD,
        model_uri=MODEL_URI,
    )
