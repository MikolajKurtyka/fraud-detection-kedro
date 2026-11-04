"""Nodes for the production pipeline: the chosen config, a holdout cut from valid, its report.

Training, evaluation, refit, submission and registration reuse the research nodes; only
the feature selection, the holdout split and the holdout report are new here.
"""

import logging
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.pipelines.evaluation.nodes import _at_threshold
from fraud_detection.pipelines.training.model import FraudModel

logger = logging.getLogger(__name__)


def select_features(feature_importance: pd.DataFrame, n_features: int | None) -> list[str]:
    """The top ``n_features`` of the research model's importance; all of them when None."""
    ranked = feature_importance.sort_values("importance", ascending=False, kind="stable")
    features = ranked["feature"].tolist()
    if n_features is not None:
        features = features[:n_features]
    logger.info("Production features: %d of %d", len(features), len(ranked))
    return features


def subset_columns(
    df: pd.DataFrame, features: list[str], target: str, join_key: str
) -> pd.DataFrame:
    """ID + target + the chosen features, for fitting (a model learns every column it gets).

    Valid, holdout and test are not cut: ``predict_proba`` selects the model's own columns,
    and the cost metrics need TransactionAmt even when it is not a feature.
    """
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise KeyError(f"Production features missing from model input: {missing[:10]}")
    return df[[join_key, target, *features]]


def split_holdout(
    valid: pd.DataFrame, holdout_fraction: float, join_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier part of valid for early stopping and the threshold, later part as holdout.

    Ordered by the join key (TransactionID grows with time), as in the time-series CV.
    """
    ordered = valid.sort_values(join_key, kind="stable")
    cut = len(ordered) - round(len(ordered) * holdout_fraction)
    prod_valid, holdout = ordered.iloc[:cut], ordered.iloc[cut:]
    logger.info("Production split of valid: %d rows valid, %d rows holdout", cut, len(holdout))
    return prod_valid, holdout


def production_options(
    model_options: dict, research_params: dict, production_params: dict
) -> tuple[dict, dict]:
    """Research model_options and tuned params, plus ``param_overrides``, under its own run."""
    model_type = model_options["model_type"]
    options = {**model_options, "run_name": f"production-{model_type}", "stage": "production"}
    params = {**research_params, **(production_params.get("param_overrides") or {})}
    return options, params


def _single_row_latency_ms(model: FraudModel, X: pd.DataFrame, n_rows: int) -> dict:
    """p50 / p99 of one-row ``predict_proba`` (the model alone, no feature building)."""
    rows = X.iloc[: max(1, min(n_rows, len(X)))]
    model.predict_proba(rows.iloc[[0]])  # warm-up
    timings = []
    for i in range(len(rows)):
        start = time.perf_counter()
        model.predict_proba(rows.iloc[[i]])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p99_ms": float(np.percentile(timings, 99)),
    }


def evaluate_holdout(
    model: FraudModel,
    holdout: pd.DataFrame,
    evaluation: dict,
    eval_params: dict,
    production_params: dict,
    serving_params: dict,
    target: str,
    join_key: str,
) -> dict:
    """Honest numbers: ranking metrics and cost on rows nothing was fitted or picked on.

    The threshold is the one picked on valid (``serving.threshold_metric``), so the holdout
    cost is what the API would really pay. Returns the valid evaluation plus ``holdout_*``
    keys, latency and feature count, logged into the production MLflow run.
    """
    y = holdout[target].to_numpy()
    X = holdout.drop(columns=[join_key, target])
    proba = model.predict_proba(X)
    threshold = float(evaluation[serving_params["threshold_metric"]])

    cost = eval_params["cost"]
    amount = holdout[cost["amount_column"]].to_numpy(dtype="float64")
    flagged = proba >= threshold
    total = (
        amount[(y == 1) & ~flagged].sum()
        + cost["review_cost"] * flagged.sum()
        + cost["false_block_cost"] * (flagged & (y == 0)).sum()
    )
    per_1k = 1000 / len(y)
    holdout_metrics = {
        "holdout_roc_auc": float(roc_auc_score(y, proba)),
        "holdout_pr_auc": float(average_precision_score(y, proba)),
        "holdout_fraud_rate": float(y.mean()),
        "holdout_rows": len(y),
        **_at_threshold(y, proba, threshold, "holdout"),
        "holdout_cost_per_1k": float(total * per_1k),
        "holdout_no_model_cost_per_1k": float(amount[y == 1].sum() * per_1k),
        "n_features": len(model.feature_columns),
        **_single_row_latency_ms(model, X, production_params["latency_rows"]),
    }

    if mlflow.active_run() is not None and model.mlflow_run_id is not None:
        client = MlflowClient()
        for key, value in holdout_metrics.items():
            client.log_metric(model.mlflow_run_id, key, value)

    logger.info(
        "%s holdout (%d rows): ROC-AUC %.4f PR-AUC %.4f | @%.3f: precision %.3f recall %.3f, "
        "cost %.0f per 1k (no model %.0f) | %d features, one row p50 %.2f ms p99 %.2f ms",
        model.model_type,
        len(y),
        holdout_metrics["holdout_roc_auc"],
        holdout_metrics["holdout_pr_auc"],
        threshold,
        holdout_metrics["holdout_precision"],
        holdout_metrics["holdout_recall"],
        holdout_metrics["holdout_cost_per_1k"],
        holdout_metrics["holdout_no_model_cost_per_1k"],
        holdout_metrics["n_features"],
        holdout_metrics["latency_p50_ms"],
        holdout_metrics["latency_p99_ms"],
    )
    return {**evaluation, **holdout_metrics}
