"""How many features does the model need? Accuracy vs size vs serving speed.

Ranks features by the saved importance of the full model (``{model}.feature_importance``),
then retrains the same model type on the top-N features for each N, with early stopping on
valid. Two hyperparameter sets per N:

- ``tuned``:   the tuned params of the full model (``{model}.best_params``)
- ``compact``: smaller / fewer trees, for the smallest / fastest model (tree models only)

For every model it records valid ROC-AUC / PR-AUC, tree count, pickle size, fit time and
three serving timings:

- ``model p50/p99``: one-row ``FraudModel.predict_proba`` (the model alone)
- ``e2e p50/p99``:   one raw row through ``InferenceBundle.predict_records`` (as the API)
- ``batch rows/s``:  ``predict_proba`` on the whole valid set

Writes ``data/08_reporting/feature_size_study_{model}.{csv,md,png}`` per model, and
``feature_size_study_all.png``: every model's saved results side by side.

Usage:
    uv run python scripts/feature_size_study.py                       # all models, default sizes
    uv run python scripts/feature_size_study.py --models lightgbm --sizes all 300 100 25
    uv run python scripts/feature_size_study.py --models xgboost catboost --configs tuned
"""

from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from kedro.framework.session import KedroSession  # noqa: E402
from kedro.framework.startup import bootstrap_project  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from fraud_detection.inference import InferenceBundle  # noqa: E402
from fraud_detection.pipelines.training.model import MODEL_TYPES, FraudModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "08_reporting"
DEFAULT_SIZES = ["all", "300", "200", "150", "100", "75", "50", "25", "10"]

# Overrides on top of the tuned params: shallow trees, fewer of them (higher learning rate).
# logreg has no compact variant.
COMPACT_PARAMS = {
    "lightgbm": {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.1},
    "xgboost": {"max_depth": 5, "learning_rate": 0.1},
    "catboost": {"depth": 5, "learning_rate": 0.1},
    "hist_gb": {"max_leaf_nodes": 15, "learning_rate": 0.1},
    "random_forest": {"n_estimators": 100, "max_depth": 12},
}

PANELS = [
    ("valid_pr_auc", "Valid PR-AUC"),
    ("model_p50_ms", "Model only, 1 row p50 (ms)"),
    ("e2e_p50_ms", "Features + model, 1 raw row p50 (ms)"),
    ("size_mb", "Model size (MB)"),
]


def percentile_ms(seconds: list[float], q: float) -> float:
    return float(np.percentile(seconds, q) * 1000)


def time_single_rows(predict, rows: list[pd.DataFrame]) -> list[float]:
    predict(rows[0])  # warm-up
    timings = []
    for row in rows:
        start = time.perf_counter()
        predict(row)
        timings.append(time.perf_counter() - start)
    return timings


def load_raw_sample(n_rows: int) -> pd.DataFrame:
    """First ``n_rows`` raw train transactions with their identity columns, as the API gets them."""
    tx = pd.read_csv(ROOT / "data/01_raw/train_transaction.csv", nrows=n_rows)
    identity = pd.read_csv(ROOT / "data/01_raw/train_identity.csv")
    identity.columns = [c.replace("-", "_") for c in identity.columns]
    return tx.merge(identity, on="TransactionID", how="left").drop(columns="isFraud")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODEL_TYPES), choices=MODEL_TYPES)
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, help="'all' or ints")
    parser.add_argument("--configs", nargs="+", default=["tuned", "compact"])
    parser.add_argument("--latency-rows", type=int, default=300, help="rows for model timing")
    parser.add_argument("--e2e-rows", type=int, default=50, help="raw rows for e2e timing")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    bootstrap_project(ROOT)
    with KedroSession.create(project_path=ROOT) as session:
        context = session.load_context()
        catalog, params = context.catalog, context.params
        train = catalog.load("mdl_in_train")
        valid = catalog.load("mdl_in_valid")
        importances = {m: catalog.load(f"{m}.feature_importance") for m in args.models}
        best_params = {m: catalog.load(f"{m}.best_params") for m in args.models}
        column_filter = catalog.load("column_filter")
        feature_encoders = catalog.load("feature_encoders")

    target, join_key = params["target"], params["feature_engineering"]["join_key"]
    X, y = train.drop(columns=[join_key, target]), train[target]
    X_valid, y_valid = valid.drop(columns=[join_key, target]), valid[target]
    n_total = X.shape[1]
    sizes = sorted(
        {n_total if s == "all" else min(int(s), n_total) for s in args.sizes}, reverse=True
    )

    raw = load_raw_sample(args.e2e_rows)
    # One-row requests as the API passes them to the model: a list with one dict, nulls left out
    raw_rows = [
        [{k: v for k, v in r.items() if not pd.isna(v)}]
        for r in raw.astype(object).to_dict("records")
    ]
    valid_rows = [X_valid.iloc[[i]] for i in range(args.latency_rows)]

    for model_type in args.models:
        ranked = importances[model_type]["feature"].tolist()
        options = dict(params[model_type]["model_options"], log_every=0)
        configs = [c for c in args.configs if c == "tuned" or model_type in COMPACT_PARAMS]
        results = []
        for config in configs:
            model_params = dict(best_params[model_type])
            if config == "compact":
                model_params.update(COMPACT_PARAMS[model_type])
            for n in sizes:
                features = ranked[:n]
                start = time.perf_counter()
                model = FraudModel(model_type, model_params, options).fit(
                    X[features], y, X_valid[features], y_valid
                )
                fit_seconds = time.perf_counter() - start
                proba = model.predict_proba(X_valid)

                start = time.perf_counter()
                model.predict_proba(X_valid)
                batch_seconds = time.perf_counter() - start

                bundle = InferenceBundle(
                    model=model,
                    column_filter=column_filter,
                    feature_encoders=feature_encoders,
                    ingestion_params=params["data_ingestion"],
                    feature_params=params["feature_engineering"],
                    target=target,
                )
                model_times = time_single_rows(model.predict_proba, valid_rows)
                e2e_times = time_single_rows(bundle.predict_records, raw_rows)

                row = {
                    "config": config,
                    "n_features": n,
                    "valid_roc_auc": roc_auc_score(y_valid, proba),
                    "valid_pr_auc": average_precision_score(y_valid, proba),
                    "trees": model.best_iteration,
                    "size_mb": len(pickle.dumps(model)) / 1e6,
                    "fit_s": fit_seconds,
                    "model_p50_ms": percentile_ms(model_times, 50),
                    "model_p99_ms": percentile_ms(model_times, 99),
                    "e2e_p50_ms": percentile_ms(e2e_times, 50),
                    "e2e_p99_ms": percentile_ms(e2e_times, 99),
                    "batch_rows_per_s": len(X_valid) / batch_seconds,
                }
                results.append(row)
                # print, not logging: Kedro's logging config hides this script's log lines
                print(
                    f"{model_type} {config} top-{n}: PR-AUC {row['valid_pr_auc']:.4f} "
                    f"ROC-AUC {row['valid_roc_auc']:.4f} | {row['trees']} trees, "
                    f"{row['size_mb']:.1f} MB | model p50 {row['model_p50_ms']:.2f} ms, "
                    f"e2e p50 {row['e2e_p50_ms']:.0f} ms",
                    flush=True,
                )

        save(pd.DataFrame(results), model_type)
        plot_all_models()


def save(table: pd.DataFrame, model_type: str) -> None:
    stem = REPORT_DIR / f"feature_size_study_{model_type}"
    table.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(to_markdown(table.round(4)), encoding="utf-8")
    fig = plot_lines(
        table, "config", f"{model_type}: accuracy, speed and size vs number of features"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(table.round(4).to_string(index=False))
    print(f"\nSaved {stem}.{{csv,md,png}}", flush=True)


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(False)]
    return "\n".join(lines) + "\n"


def plot_lines(table: pd.DataFrame, by: str, title: str):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 4))
    for ax, (col, panel_title) in zip(axes, PANELS, strict=True):
        for label, group in table.groupby(by):
            ax.plot(group["n_features"], group[col], marker="o", label=label)
        ax.set_title(panel_title)
        ax.set_xlabel("Features (top-N by importance)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    return fig


def plot_all_models() -> None:
    """Tuned configs of every model with saved results, one line per model."""
    tables = [
        pd.read_csv(path).assign(model=path.stem.removeprefix("feature_size_study_"))
        for path in sorted(REPORT_DIR.glob("feature_size_study_*.csv"))
    ]
    tuned = pd.concat(tables).query("config == 'tuned'")
    path = REPORT_DIR / "feature_size_study_all.png"
    fig = plot_lines(tuned, "model", "All models (tuned params) vs number of features")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
