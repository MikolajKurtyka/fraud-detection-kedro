"""Nodes for the serving pipeline: package a trained model for the API and register it."""

import logging
import pickle
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fraud_detection.inference import FraudPyfunc, InferenceBundle
from fraud_detection.pipelines.training.model import FraudModel

logger = logging.getLogger(__name__)


def _champion_score(client: MlflowClient, name: str, alias: str, metric: str) -> float | None:
    """The promotion metric of the version currently behind the alias; None if there is none."""
    try:
        champion = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None
    value = champion.tags.get(metric)
    return float(value) if value is not None else None


def register_model(
    model: FraudModel,
    evaluation: dict,
    column_filter: dict,
    feature_encoders: dict,
    ingestion_params: dict,
    feature_params: dict,
    target: str,
    params: dict,
) -> dict:
    """Log the full inference chain as one MLflow pyfunc model, register it, and move the
    alias to it if it beats the current champion on valid.

    The threshold and metrics come from the evaluation of this exact (train-only) model, and
    are stored with it, so the API never pairs a model with another model's threshold.
    """
    name, alias, metric = (
        params["registered_model_name"],
        params["alias"],
        params["promotion_metric"],
    )
    threshold = float(evaluation[params["threshold_metric"]])
    score = float(evaluation[metric])
    summary = {
        "model_type": model.model_type,
        "threshold_metric": params["threshold_metric"],
        "threshold": threshold,
        metric: score,
        "registered_model_name": name,
        "version": None,
        "promoted": False,
    }
    # Only inside a kedro run (kedro-mlflow has a run open); tests register nothing.
    if mlflow.active_run() is None:
        logger.warning("No active MLflow run, %s is not registered", model.model_type)
        return summary

    bundle = InferenceBundle(
        model=model,
        column_filter=column_filter,
        feature_encoders=feature_encoders,
        ingestion_params=ingestion_params,
        feature_params=feature_params,
        target=target,
    )
    metadata = {"model_type": model.model_type, "threshold": threshold, metric: score}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.pkl"
        path.write_bytes(pickle.dumps(bundle))
        info = mlflow.pyfunc.log_model(
            name="serving_model",
            python_model=FraudPyfunc(),
            artifacts={"bundle": str(path)},
            metadata=metadata,
            registered_model_name=name,
        )

    version = str(info.registered_model_version)
    client = MlflowClient()
    # Older pickles do not store the training run
    tags = {**metadata, "training_run_id": getattr(model, "mlflow_run_id", None)}
    for key, value in tags.items():
        if value is not None:
            client.set_model_version_tag(name, version, key, str(value))

    champion = _champion_score(client, name, alias, metric)
    promoted = champion is None or score > champion
    if promoted:
        client.set_registered_model_alias(name, alias, version)
    logger.info(
        "Registered %s v%s (%s, %s %.4f, threshold %.4f); %s",
        name,
        version,
        model.model_type,
        metric,
        score,
        threshold,
        f"@{alias} -> v{version}"
        if promoted
        else f"@{alias} kept ({metric} {champion:.4f} >= {score:.4f})",
    )
    return {**summary, "version": version, "promoted": promoted, "champion_" + metric: champion}
