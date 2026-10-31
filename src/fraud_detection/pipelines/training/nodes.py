"""Nodes for the training pipeline: tuning, final fit, time-series CV and refit, in MLflow.

kedro-mlflow opens one MLflow run per ``kedro run``. Each model logs into its own
nested child run (same metric names for every model, so they compare side by side
in the MLflow UI); Optuna trials are nested one level deeper. CV and refit numbers
are added to the model's final run.
"""

import logging
import time
from contextlib import contextmanager

import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .model import FraudModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
_PERMUTATION_ROWS = 20_000


@contextmanager
def _child_run(name: str, **tags):
    """Nested MLflow run under the kedro-mlflow run; a no-op when no run is active (tests)."""
    if mlflow.active_run() is None:
        yield None
        return
    with mlflow.start_run(run_name=name, nested=True, tags=tags) as run:
        yield run


def _split_xy(df: pd.DataFrame, target: str, join_key: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[join_key, target]), df[target]


def _score(y: pd.Series, proba) -> dict[str, float]:
    return {name: float(fn(y, proba)) for name, fn in _METRICS.items()}


def _log_to_training_run(model: FraudModel, metrics: dict) -> None:
    """Add metrics to the model's final run; a no-op outside a kedro run (tests)."""
    if mlflow.active_run() is None or model.mlflow_run_id is None:
        return
    client = MlflowClient()
    for key, value in metrics.items():
        client.log_metric(model.mlflow_run_id, key, value)


def _permutation_importance(model: FraudModel, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Drop in valid PR-AUC when one column is shuffled, on a sample of valid rows.

    Used for models without built-in importance (logistic regression, HistGradientBoosting).
    """
    rng = np.random.default_rng(42)
    if len(X) > _PERMUTATION_ROWS:
        rows = rng.choice(len(X), _PERMUTATION_ROWS, replace=False)
        X, y = X.iloc[rows], y.iloc[rows]
    X = X.copy()
    base = average_precision_score(y, model.predict_proba(X))
    drops = []
    for col in model.feature_columns:
        original = X[col]
        # Positional shuffle; set_axis keeps the dtype (category columns stay category)
        X[col] = original.iloc[rng.permutation(len(X))].set_axis(X.index)
        drops.append(base - average_precision_score(y, model.predict_proba(X)))
        X[col] = original
    return (
        pd.DataFrame({"feature": model.feature_columns, "importance": drops})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _suggest(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            raise ValueError(f"Unknown search space type {spec['type']!r} for {name}")
    return params


def tune_hyperparameters(
    train: pd.DataFrame, valid: pd.DataFrame, options: dict, target: str, join_key: str
) -> dict:
    """Optuna search over ``search_space``; returns the full parameter set to train with.

    With ``n_trials: 0`` the base ``params`` are returned unchanged.
    """
    model_type, base = options["model_type"], options["params"]
    if options["n_trials"] == 0:
        logger.info("%s: tuning skipped (n_trials=0), using base params", model_type)
        return dict(base)

    X, y = _split_xy(train, target, join_key)
    X_valid, y_valid = _split_xy(valid, target, join_key)
    metric = options["tuning_metric"]

    with _child_run(f"{model_type}-tuning", model_type=model_type, stage="tuning") as tuning_run:

        def objective(trial: optuna.Trial) -> float:
            params = {**base, **_suggest(trial, options["search_space"])}
            start = time.time()
            model = FraudModel(model_type, params, options).fit(X, y, X_valid, y_valid)
            scores = _score(y_valid, model.predict_proba(X_valid))
            trial.set_user_attr("best_iteration", model.best_iteration)
            logger.info(
                "%s trial %d: valid %s %.5f (best iteration %s, %.0fs)",
                model_type,
                trial.number,
                metric,
                scores[metric],
                model.best_iteration,
                time.time() - start,
            )
            with _child_run(f"{model_type}-trial-{trial.number}", stage="trial") as run:
                if run is not None:
                    mlflow.log_params(params)
                    mlflow.log_metrics({f"valid_{k}": v for k, v in scores.items()})
                    if model.best_iteration is not None:
                        mlflow.log_metric("best_iteration", model.best_iteration)
            return scores[metric]

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            objective, n_trials=options["n_trials"], timeout=options["timeout_minutes"] * 60
        )
        if tuning_run is not None:
            mlflow.log_params(study.best_params)
            mlflow.log_metric(f"best_valid_{metric}", study.best_value)

    logger.info(
        "%s: best valid %s %.5f after %d trials, params %s",
        model_type,
        metric,
        study.best_value,
        len(study.trials),
        study.best_params,
    )
    return {**base, **study.best_params}


def train_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    params: dict,
    options: dict,
    target: str,
    join_key: str,
) -> tuple[FraudModel, dict, pd.DataFrame]:
    """Fit the final model with early stopping on valid; report train and valid metrics.

    A large train-valid gap means overfitting or features that drift over time.
    """
    model_type = options["model_type"]
    X, y = _split_xy(train, target, join_key)
    X_valid, y_valid = _split_xy(valid, target, join_key)

    with _child_run(model_type, model_type=model_type, stage="final") as run:
        start = time.time()
        model = FraudModel(model_type, params, options).fit(X, y, X_valid, y_valid)
        fit_seconds = time.time() - start

        train_scores = _score(y, model.predict_proba(X))
        valid_scores = _score(y_valid, model.predict_proba(X_valid))
        metrics = {
            **{f"train_{k}": v for k, v in train_scores.items()},
            **{f"valid_{k}": v for k, v in valid_scores.items()},
            "roc_auc_gap": train_scores["roc_auc"] - valid_scores["roc_auc"],
            "best_iteration": model.best_iteration,
            "n_features": len(model.feature_columns),
            "fit_seconds": round(fit_seconds, 1),
        }
        importance = model.feature_importance()
        if importance is None:
            importance = _permutation_importance(model, X_valid, y_valid)

        if run is not None:
            model.mlflow_run_id = run.info.run_id
            mlflow.log_params(
                {
                    **params,
                    "device": options.get("device", "cpu"),
                    "imbalance": model.imbalance["method"],
                }
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")

    logger.info(
        "%s: valid ROC-AUC %.5f, PR-AUC %.5f | train ROC-AUC %.5f (gap %.4f) | "
        "best iteration %s, %.0fs",
        model_type,
        metrics["valid_roc_auc"],
        metrics["valid_pr_auc"],
        metrics["train_roc_auc"],
        metrics["roc_auc_gap"],
        model.best_iteration,
        fit_seconds,
    )
    return model, metrics, importance


def cross_validate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model: FraudModel,
    params: dict,
    options: dict,
    target: str,
    join_key: str,
) -> dict:
    """Expanding-window time-series CV over train + valid; off unless ``cv_folds`` > 0.

    Rows are ordered by the join key (TransactionID grows with time in this data), and each
    fold trains on all rows before its valid block. Gives mean +- std instead of the single
    number from one holdout. Early stopping uses each fold's valid block, as in train_model.
    """
    model_type, n_folds = options["model_type"], options.get("cv_folds", 0)
    if n_folds == 0:
        logger.info("%s: time-series CV skipped (cv_folds=0)", model_type)
        return {"cv_folds": 0}

    full = pd.concat([train, valid]).sort_values(join_key, kind="stable")
    X, y = _split_xy(full, target, join_key)
    folds = []
    for fold, (fit_rows, valid_rows) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(X)):
        X_fit, y_fit = X.iloc[fit_rows], y.iloc[fit_rows]
        X_val, y_val = X.iloc[valid_rows], y.iloc[valid_rows]
        fold_model = FraudModel(model_type, params, options).fit(X_fit, y_fit, X_val, y_val)
        scores = _score(y_val, fold_model.predict_proba(X_val))
        folds.append(
            {
                "fold": fold,
                "train_rows": len(fit_rows),
                "valid_rows": len(valid_rows),
                **scores,
                "best_iteration": fold_model.best_iteration,
            }
        )
        logger.info(
            "%s CV fold %d/%d: train %d rows, valid %d rows, ROC-AUC %.5f, PR-AUC %.5f",
            model_type,
            fold + 1,
            n_folds,
            len(fit_rows),
            len(valid_rows),
            scores["roc_auc"],
            scores["pr_auc"],
        )

    summary = {"cv_folds": n_folds}
    for metric in _METRICS:
        values = [f[metric] for f in folds]
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_std"] = float(np.std(values))
    _log_to_training_run(model, summary)
    logger.info(
        "%s CV (%d folds): ROC-AUC %.5f +- %.5f, PR-AUC %.5f +- %.5f",
        model_type,
        n_folds,
        summary["cv_roc_auc_mean"],
        summary["cv_roc_auc_std"],
        summary["cv_pr_auc_mean"],
        summary["cv_pr_auc_std"],
    )
    return {**summary, "folds": folds}


def refit_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model: FraudModel,
    params: dict,
    options: dict,
    target: str,
    join_key: str,
) -> FraudModel:
    """Refit on train + valid for the test submission; off when ``refit`` is false.

    The valid days are the closest to test, so the submission model should learn from them.
    No data is left for early stopping, so iterative models build a fixed number of trees:
    ``best_iteration`` of the train-only model times ``refit_iteration_multiplier`` (more data
    supports more trees). The refitted model has no honest score of its own; valid metrics
    always come from the train-only model.
    """
    model_type = options["model_type"]
    if not options.get("refit", False):
        logger.info("%s: refit off, the submission uses the train-only model", model_type)
        return model

    X, y = _split_xy(pd.concat([train, valid]), target, join_key)
    n_iterations = None
    if model.best_iteration is not None:
        n_iterations = max(1, round(model.best_iteration * options["refit_iteration_multiplier"]))

    start = time.time()
    refitted = FraudModel(model_type, params, options).fit(X, y, n_iterations=n_iterations)
    fit_seconds = time.time() - start
    refitted.mlflow_run_id = model.mlflow_run_id

    refit_metrics = {"refit_rows": len(X), "refit_seconds": round(fit_seconds, 1)}
    if n_iterations is not None:
        refit_metrics["refit_iterations"] = n_iterations
    _log_to_training_run(model, refit_metrics)
    logger.info(
        "%s: refitted on train + valid (%d rows, %s trees), %.0fs",
        model_type,
        len(X),
        n_iterations if n_iterations is not None else "n/a",
        fit_seconds,
    )
    return refitted
