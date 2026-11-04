"""Nodes for the production pipeline: candidates, a holdout cut from valid, the choice.

Each candidate (model type + top-N features) reuses the research training, evaluation, refit
and submission nodes; new here are the feature selection, the holdout split and report, and
the choice of one candidate for the API and Kaggle.
"""

import logging
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.pipelines.evaluation.nodes import _at_threshold
from fraud_detection.pipelines.serving.nodes import register_model
from fraud_detection.pipelines.training.model import FraudModel

logger = logging.getLogger(__name__)


def select_features(feature_importance: pd.DataFrame, candidate: dict) -> list[str]:
    """The top ``candidate["n_features"]`` of the research model's importance (None = all)."""
    n_features = candidate.get("n_features")
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
    model_options: dict, research_params: dict, production_params: dict, candidate: dict
) -> tuple[dict, dict]:
    """Research model_options and tuned params, plus the global and the candidate's
    ``param_overrides``, under its own MLflow run (production-<model>-<n_features>)."""
    model_type = model_options["model_type"]
    size = candidate.get("n_features") or "all"
    options = {
        **model_options,
        "run_name": f"production-{model_type}-{size}",
        "stage": "production",
    }
    params = {
        **research_params,
        **(production_params.get("param_overrides") or {}),
        **(candidate.get("param_overrides") or {}),
    }
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


# Report column -> number format
_REPORT = {
    "n_features": "{:d}",
    "holdout_pr_auc": "{:.4f}",
    "holdout_roc_auc": "{:.4f}",
    "holdout_cost_per_1k": "{:.0f}",
    "latency_p50_ms": "{:.2f}",
    "latency_p99_ms": "{:.2f}",
}


def choose_candidate(candidates: dict, selection: dict, **evaluations: dict) -> tuple[dict, str]:
    """Pick the API + Kaggle model among the candidates, on their holdout reports.

    1. Keep the candidates within the latency budget (one-row p99 <= max_latency_p99_ms;
       null = no budget). If none is, keep all of them and warn.
    2. Take the best holdout PR-AUC among them; every candidate within pr_auc_tolerance of
       it counts as equally good.
    3. Of those, the fewest features (then the higher PR-AUC).

    Returns the choice and a Markdown table of every candidate.
    """
    table = pd.DataFrame(
        [
            {
                "candidate": name,
                "model": candidates[name]["model"],
                "n_features": int(ev["n_features"]),
                **{key: float(ev[key]) for key in list(_REPORT)[1:]},
            }
            for name, ev in evaluations.items()
        ]
    )
    budget, tolerance = selection.get("max_latency_p99_ms"), selection["pr_auc_tolerance"]
    table["fast_enough"] = True if budget is None else table["latency_p99_ms"] <= budget
    pool = table[table["fast_enough"]]
    if pool.empty:
        logger.warning("No candidate has one-row p99 <= %s ms; choosing among all", budget)
        pool = table
    best = pool.loc[pool["holdout_pr_auc"].idxmax()]
    near = pool[pool["holdout_pr_auc"] >= best["holdout_pr_auc"] - tolerance]
    chosen = near.sort_values(
        ["n_features", "holdout_pr_auc"], ascending=[True, False], kind="stable"
    ).iloc[0]

    choice = {
        "candidate": str(chosen["candidate"]),
        "model": str(chosen["model"]),
        "n_features": int(chosen["n_features"]),
        "holdout_pr_auc": float(chosen["holdout_pr_auc"]),
        "latency_p99_ms": float(chosen["latency_p99_ms"]),
        "best_candidate": str(best["candidate"]),
        "best_holdout_pr_auc": float(best["holdout_pr_auc"]),
        "max_latency_p99_ms": budget,
        "pr_auc_tolerance": tolerance,
        "any_fast_enough": bool(table["fast_enough"].any()),
    }
    logger.info(
        "Production choice: %s (%d features, holdout PR-AUC %.4f, p99 %.2f ms); "
        "best PR-AUC: %s %.4f",
        choice["candidate"],
        choice["n_features"],
        choice["holdout_pr_auc"],
        choice["latency_p99_ms"],
        choice["best_candidate"],
        choice["best_holdout_pr_auc"],
    )
    return choice, _candidates_markdown(table, choice)


def _candidates_markdown(table: pd.DataFrame, choice: dict) -> str:
    budget = choice["max_latency_p99_ms"]
    columns = ["candidate", "model", *_REPORT, "fast_enough", "chosen"]
    lines = [
        "# Production candidates",
        "",
        "Holdout = the later part of valid: nothing was fitted, early-stopped or picked on it.",
        f"Rule: one-row p99 <= {'any' if budget is None else budget} ms, then the best holdout "
        f"PR-AUC; of the candidates within {choice['pr_auc_tolerance']} of it, the fewest "
        "features.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in table.sort_values("holdout_pr_auc", ascending=False).to_dict("records"):
        cells = [row["candidate"], row["model"]]
        cells += [fmt.format(row[key]) for key, fmt in _REPORT.items()]
        cells += [
            "yes" if row["fast_enough"] else "no",
            "**yes**" if row["candidate"] == choice["candidate"] else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not choice["any_fast_enough"]:
        lines += ["", f"No candidate met the {budget} ms budget; chosen among all of them."]
    return "\n".join(lines) + "\n"


def register_chosen(
    choice: dict,
    column_filter: dict,
    feature_encoders: dict,
    ingestion_params: dict,
    feature_params: dict,
    target: str,
    serving_params: dict,
    validation_params: dict,
    **candidates: object,
) -> dict:
    """Register the chosen candidate's train-only model; @champion moves only if it beats
    the current champion. ``candidates`` holds ``<name>__model`` and ``<name>__evaluation``."""
    name = choice["candidate"]
    return register_model(
        candidates[f"{name}__model"],
        candidates[f"{name}__evaluation"],
        column_filter,
        feature_encoders,
        ingestion_params,
        feature_params,
        target,
        serving_params,
        validation_params,
        extra_metadata={"candidate": name, "n_features": choice["n_features"]},
    )


def copy_submission(choice: dict, **submissions: pd.DataFrame) -> pd.DataFrame:
    """The chosen candidate's Kaggle file, saved again as submission_production.csv."""
    return submissions[choice["candidate"]]
