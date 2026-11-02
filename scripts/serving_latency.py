"""Single-row serving latency of every saved model: scoring paths and thread settings.

For each ``data/06_models/{model}.pkl`` (the train-only models, all features), times one row
at a time (p50 in ms, after one warm-up call):

- ``model_threads_ms``:     ``FraudModel.predict_proba`` on model input, default threads
- ``e2e_pandas_ms``:        a raw row through ``InferenceBundle.predict_proba`` (pandas path)
- ``e2e_numpy_ms``:         a raw row through ``InferenceBundle.predict_records`` (the API's path)
- ``model_1thread_ms`` / ``e2e_numpy_1thread_ms``: the same with one thread

Writes ``data/08_reporting/serving_latency.csv``.

Usage:
    uv run python scripts/serving_latency.py
    uv run python scripts/serving_latency.py --models lightgbm xgboost
"""

from __future__ import annotations

import argparse
import functools
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from feature_size_study import REPORT_DIR, ROOT, load_raw_sample
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from threadpoolctl import threadpool_limits

from fraud_detection.inference import InferenceBundle
from fraud_detection.pipelines.training.model import MODEL_TYPES, FraudModel


def p50_ms(predict, rows: list) -> float:
    predict(rows[0])  # warm-up
    timings = []
    for row in rows:
        start = time.perf_counter()
        predict(row)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings) * 1000)


def use_one_thread(model: FraudModel) -> None:
    """Each library's own thread setting; OpenMP / BLAS pools are limited by the caller."""
    estimator = model.estimator
    if model.model_type in ("lightgbm", "xgboost", "random_forest"):
        estimator.set_params(n_jobs=1)
    elif model.model_type == "catboost":  # a fitted CatBoost model takes it per call only
        estimator.predict_proba = functools.partial(estimator.predict_proba, thread_count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODEL_TYPES), choices=MODEL_TYPES)
    parser.add_argument("--model-rows", type=int, default=300, help="rows for model timing")
    parser.add_argument("--e2e-rows", type=int, default=50, help="raw rows for e2e timing")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    bootstrap_project(ROOT)
    with KedroSession.create(project_path=ROOT) as session:
        context = session.load_context()
        catalog, params = context.catalog, context.params
        valid = catalog.load("mdl_in_valid")
        column_filter = catalog.load("column_filter")
        feature_encoders = catalog.load("feature_encoders")

    join_key = params["feature_engineering"]["join_key"]
    X_valid = valid.drop(columns=[join_key, params["target"]])
    model_rows = [X_valid.iloc[[i]] for i in range(args.model_rows)]
    raw = load_raw_sample(args.e2e_rows)
    raw_frames = [raw.iloc[[i]] for i in range(len(raw))]
    # As the API sends them: a list with one dict, nulls left out
    raw_records = [
        [{k: v for k, v in r.items() if not pd.isna(v)}]
        for r in raw.astype(object).to_dict("records")
    ]

    results = []
    for model_type in args.models:
        with open(ROOT / "data" / "06_models" / f"{model_type}.pkl", "rb") as f:
            model: FraudModel = pickle.load(f)
        bundle = InferenceBundle(
            model=model,
            column_filter=column_filter,
            feature_encoders=feature_encoders,
            ingestion_params=params["data_ingestion"],
            feature_params=params["feature_engineering"],
            target=params["target"],
        )
        row = {
            "model": model_type,
            "model_threads_ms": p50_ms(model.predict_proba, model_rows),
            "e2e_pandas_ms": p50_ms(bundle.predict_proba, raw_frames),
            "e2e_numpy_ms": p50_ms(bundle.predict_records, raw_records),
        }
        use_one_thread(model)
        with threadpool_limits(1):
            row["model_1thread_ms"] = p50_ms(model.predict_proba, model_rows)
            row["e2e_numpy_1thread_ms"] = p50_ms(bundle.predict_records, raw_records)
        results.append(row)
        print(", ".join(f"{k}={v:.2f}" if isinstance(v, float) else v for k, v in row.items()))

    table = pd.DataFrame(results).round(2)
    path = REPORT_DIR / "serving_latency.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
