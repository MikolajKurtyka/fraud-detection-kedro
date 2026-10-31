"""Nodes for the evaluation pipeline: threshold metrics, curves, Kaggle submissions.

Metrics are also written into the model's MLflow training run, so one row in the
MLflow UI holds both training and evaluation numbers for that model.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are only saved

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraud_detection.pipelines.training.model import FraudModel

logger = logging.getLogger(__name__)


def _at_threshold(y: np.ndarray, proba: np.ndarray, threshold: float, prefix: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, proba >= threshold, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tn": int(tn),
    }


def _training_run_id(model: FraudModel) -> str | None:
    """The MLflow run the model was trained in.

    Older pickles do not store it; then the latest final run of that model type is used,
    which is the run that wrote the current pickle.
    """
    run_id = getattr(model, "mlflow_run_id", None)
    if run_id is not None:
        return run_id
    runs = mlflow.search_runs(
        experiment_names=["fraud-detection"],
        filter_string=f"tags.stage = 'final' and tags.model_type = '{model.model_type}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return None if runs.empty else runs["run_id"].iloc[0]


def _plot_curves(y: np.ndarray, proba: np.ndarray, metrics: dict, model_type: str):
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    precision, recall, _ = precision_recall_curve(y, proba)
    ax_pr.plot(recall, precision, color="#2563eb")
    ax_pr.scatter(
        metrics["best_f1_recall"],
        metrics["best_f1_precision"],
        color="#dc2626",
        zorder=3,
        label=f"best F1 (threshold {metrics['best_f1_threshold']:.3f})",
    )
    ax_pr.axhline(y.mean(), color="#9ca3af", linestyle="--", label=f"base rate {y.mean():.3f}")
    ax_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title=f"Precision-recall (PR-AUC {metrics['pr_auc']:.4f})",
    )
    ax_pr.legend(loc="upper right")

    fpr, tpr, _ = roc_curve(y, proba)
    ax_roc.plot(fpr, tpr, color="#2563eb")
    ax_roc.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
    ax_roc.scatter(
        metrics["at_fpr_fpr"],
        metrics["at_fpr_recall"],
        color="#dc2626",
        zorder=3,
        label=f"recall {metrics['at_fpr_recall']:.3f} at FPR {metrics['at_fpr_fpr']:.3f}",
    )
    ax_roc.set(
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title=f"ROC (ROC-AUC {metrics['roc_auc']:.4f})",
    )
    ax_roc.legend(loc="lower right")

    fig.suptitle(f"{model_type} on valid")
    fig.tight_layout()
    return fig


def evaluate_model(
    model: FraudModel, valid: pd.DataFrame, params: dict, target: str, join_key: str
) -> tuple[dict, plt.Figure]:
    """Ranking metrics plus precision / recall / F1 at three thresholds, on valid."""
    y = valid[target].to_numpy()
    proba = model.predict_proba(valid.drop(columns=[join_key, target]))

    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_f1_threshold = thresholds[np.argmax(f1[:-1])]

    fpr, _, roc_thresholds = roc_curve(y, proba)
    at_fpr_threshold = roc_thresholds[np.searchsorted(fpr, params["target_fpr"], side="right") - 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "fraud_rate": float(y.mean()),
        **_at_threshold(y, proba, params["fixed_threshold"], "fixed"),
        **_at_threshold(y, proba, best_f1_threshold, "best_f1"),
        **_at_threshold(y, proba, at_fpr_threshold, "at_fpr"),
    }
    figure = _plot_curves(y, proba, metrics, model.model_type)

    # Only inside a kedro run (kedro-mlflow has a run open); tests log nothing.
    run_id = _training_run_id(model) if mlflow.active_run() else None
    if run_id is not None:
        client = MlflowClient()
        for key, value in metrics.items():
            client.log_metric(run_id, f"eval_{key}", value)
        client.log_figure(run_id, figure, "evaluation_curves.png")

    logger.info(
        "%s valid: ROC-AUC %.4f PR-AUC %.4f | @%.2f: precision %.3f recall %.3f | "
        "best F1 %.3f @%.3f: precision %.3f recall %.3f | recall %.3f @ FPR %.3f%s",
        model.model_type,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["fixed_threshold"],
        metrics["fixed_precision"],
        metrics["fixed_recall"],
        metrics["best_f1_f1"],
        metrics["best_f1_threshold"],
        metrics["best_f1_precision"],
        metrics["best_f1_recall"],
        metrics["at_fpr_recall"],
        metrics["at_fpr_fpr"],
        f" | logged to MLflow run {run_id[:8]}" if run_id else "",
    )
    return metrics, figure


def predict_submission(model: FraudModel, test: pd.DataFrame, join_key: str) -> pd.DataFrame:
    """Kaggle submission: TransactionID, isFraud probability for every test row."""
    proba = model.predict_proba(test.drop(columns=[join_key]))
    submission = pd.DataFrame({join_key: test[join_key].to_numpy(), "isFraud": proba})
    logger.info(
        "%s submission: %d rows, mean probability %.4f",
        model.model_type,
        len(submission),
        submission["isFraud"].mean(),
    )
    return submission


def blend_submissions(*submissions: pd.DataFrame) -> pd.DataFrame:
    """Rank-average several submissions, scaled back to 0-1."""
    base = submissions[0]
    for other in submissions[1:]:
        if not base["TransactionID"].equals(other["TransactionID"]):
            raise ValueError("Submissions do not cover the same TransactionIDs in the same order")
    ranks = np.mean([s["isFraud"].rank(pct=True).to_numpy() for s in submissions], axis=0)
    logger.info("Blended %d submissions by rank average", len(submissions))
    return pd.DataFrame({"TransactionID": base["TransactionID"], "isFraud": ranks})
