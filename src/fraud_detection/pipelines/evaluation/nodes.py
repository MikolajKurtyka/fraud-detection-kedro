"""Nodes for the evaluation pipeline: threshold and cost metrics, curves, Kaggle submissions.

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


def _costs(
    y: np.ndarray, proba: np.ndarray, amount: np.ndarray, cost: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Total cost at every candidate threshold (each distinct score, plus "flag nothing").

    Cost = missed fraud amounts + review_cost per flagged row + false_block_cost per flagged
    legitimate row. Flagging is ``proba >= threshold``, as in the API.
    """
    order = np.argsort(-proba, kind="stable")
    p, fraud = proba[order], y[order] == 1
    caught = np.cumsum(np.where(fraud, amount[order], 0.0))
    flagged = np.arange(1, len(p) + 1)
    false_alarms = np.cumsum(~fraud)
    # One candidate per distinct score: the last row of each run of equal scores
    last = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    missed = caught[-1] - caught[last]
    total = (
        missed + cost["review_cost"] * flagged[last] + cost["false_block_cost"] * false_alarms[last]
    )
    thresholds = np.r_[np.nextafter(p[0], np.inf), p[last]]
    return thresholds, np.r_[caught[-1], total]


def _cost_metrics(
    y: np.ndarray, proba: np.ndarray, amount: np.ndarray, cost: dict, thresholds: dict
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """The cost-optimal threshold on valid, the cost per 1k transactions at every reported
    threshold, and the cost curve (candidate thresholds, cost per 1k) for the plot."""
    candidates, totals = _costs(y, proba, amount, cost)
    best = int(np.argmin(totals))
    per_1k = 1000 / len(y)
    metrics = {
        **_at_threshold(y, proba, candidates[best], "min_cost"),
        "no_model_cost_per_1k": float(amount[y == 1].sum() * per_1k),
    }
    for prefix, threshold in {**thresholds, "min_cost": candidates[best]}.items():
        flagged = proba >= threshold
        total = (
            amount[(y == 1) & ~flagged].sum()
            + cost["review_cost"] * flagged.sum()
            + cost["false_block_cost"] * (flagged & (y == 0)).sum()
        )
        metrics[f"{prefix}_cost_per_1k"] = float(total * per_1k)
    return metrics, (candidates, totals * per_1k)


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


def _plot_curves(
    y: np.ndarray,
    proba: np.ndarray,
    metrics: dict,
    cost_curve: tuple[np.ndarray, np.ndarray],
    model_type: str,
):
    fig, (ax_pr, ax_roc, ax_cost) = plt.subplots(1, 3, figsize=(18, 5))

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

    thresholds, cost_per_1k = cost_curve
    in_range = thresholds <= 1
    ax_cost.plot(thresholds[in_range], cost_per_1k[in_range], color="#2563eb")
    ax_cost.axhline(
        metrics["no_model_cost_per_1k"], color="#9ca3af", linestyle="--", label="no model"
    )
    ax_cost.scatter(
        metrics["min_cost_threshold"],
        metrics["min_cost_cost_per_1k"],
        color="#dc2626",
        zorder=3,
        label=f"min cost (threshold {metrics['min_cost_threshold']:.3f})",
    )
    ax_cost.set(
        xlabel="Threshold",
        ylabel="Cost per 1k transactions",
        xlim=(0, 1),
        ylim=(0, 1.5 * metrics["no_model_cost_per_1k"]),  # flagging everything costs far more
        title=f"Expected cost (min {metrics['min_cost_cost_per_1k']:.0f} per 1k)",
    )
    ax_cost.legend(loc="upper right")

    fig.suptitle(f"{model_type} on valid")
    fig.tight_layout()
    return fig


def evaluate_model(
    model: FraudModel, valid: pd.DataFrame, params: dict, target: str, join_key: str
) -> tuple[dict, plt.Figure]:
    """Ranking metrics, precision / recall / F1 at four thresholds and their cost, on valid.

    The fourth threshold, ``min_cost``, minimises the expected cost of missed fraud, reviews
    and blocked good customers (``params["cost"]``). The serving pipeline can store it with
    the model, so the API decides by cost.
    """
    y = valid[target].to_numpy()
    proba = model.predict_proba(valid.drop(columns=[join_key, target]))

    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_f1_threshold = thresholds[np.argmax(f1[:-1])]

    fpr, _, roc_thresholds = roc_curve(y, proba)
    at_fpr_threshold = roc_thresholds[np.searchsorted(fpr, params["target_fpr"], side="right") - 1]

    thresholds = {
        "fixed": params["fixed_threshold"],
        "best_f1": best_f1_threshold,
        "at_fpr": at_fpr_threshold,
    }
    amount = valid[params["cost"]["amount_column"]].to_numpy(dtype="float64")
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "fraud_rate": float(y.mean()),
        **{
            k: v
            for prefix, threshold in thresholds.items()
            for k, v in _at_threshold(y, proba, threshold, prefix).items()
        },
    }
    cost_metrics, cost_curve = _cost_metrics(y, proba, amount, params["cost"], thresholds)
    metrics.update(cost_metrics)
    figure = _plot_curves(y, proba, metrics, cost_curve, model.model_type)

    # Only inside a kedro run (kedro-mlflow has a run open); tests log nothing.
    run_id = _training_run_id(model) if mlflow.active_run() else None
    if run_id is not None:
        client = MlflowClient()
        for key, value in metrics.items():
            client.log_metric(run_id, f"eval_{key}", value)
        client.log_figure(run_id, figure, "evaluation_curves.png")

    logger.info(
        "%s valid: ROC-AUC %.4f PR-AUC %.4f | @%.2f: precision %.3f recall %.3f | "
        "best F1 %.3f @%.3f: precision %.3f recall %.3f | recall %.3f @ FPR %.3f | "
        "min cost %.0f per 1k @%.3f (no model %.0f)%s",
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
        metrics["min_cost_cost_per_1k"],
        metrics["min_cost_threshold"],
        metrics["no_model_cost_per_1k"],
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
