"""Serving parity: raw rows through InferenceBundle must equal the batch pipelines' output.

Builds the fitted artifacts with the real nodes (ingestion -> cleaning -> features ->
training) on a small synthetic dataset, then scores the same rows the way the API sends
them: one dict per transaction, with missing keys, numbers as strings and ``id-01`` names.
Both serving paths are checked: the pandas path (``predict_proba``) and the NumPy fast path
(``predict_records``), which must give exactly the same features and scores.
"""

import pickle
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraud_detection.inference import FraudPyfunc, InferenceBundle
from fraud_detection.pipelines.data_cleaning.nodes import apply_column_filter, fit_column_filter
from fraud_detection.pipelines.data_ingestion.nodes import (
    merge_transaction_identity,
    normalize_identity_columns,
    type_identity,
    type_transaction,
)
from fraud_detection.pipelines.feature_engineering.nodes import (
    add_stateless_features,
    apply_feature_encoders,
    build_model_input,
    fit_feature_encoders,
)
from fraud_detection.pipelines.serving.nodes import register_model
from fraud_detection.pipelines.training.model import FraudModel

TARGET = "isFraud"
KEY = "TransactionID"
INGESTION = {
    "join_key": KEY,
    "schema": {
        "transaction": {
            "categorical": ["card1", "card4", "addr1", "P_emaildomain"],
            "numerical": ["TransactionDT", "TransactionAmt", "D1", "D15", "V{1..3}"],
        },
        "identity": {"categorical": ["DeviceType"], "numerical": ["id_{01..02}"]},
    },
    "numerical_dtype": "float32",
    "numerical_dtype_overrides": {"TransactionDT": "int64", "TransactionAmt": "float64"},
}
CLEANING = {
    "join_key": KEY,
    "null_threshold": 0.99,
    "top_value_threshold": 0.99,
    "v_column_pattern": r"^V\d+$",
    "v_correlation_threshold": 0.95,
    "protected_columns": [KEY, "TransactionDT", "TransactionAmt", "has_identity", "card1", "D1"],
}
FEATURES = {
    "join_key": KEY,
    "time_column": "TransactionDT",
    "d_columns": ["D1", "D15"],
    "email_columns": ["P_emaildomain"],
    "uid_columns": ["card1", "addr1", "D1_norm"],
    "min_group_size": 2,
    "group_features": [
        {"value": "TransactionAmt", "by": "card1", "stats": ["mean", "std"], "ratio": True},
        {"value": "TransactionAmt", "by": "uid", "stats": ["mean"], "ratio": False},
    ],
    "frequency_columns": ["card1", "uid"],
    "max_native_categories": 3,
    "drop_columns": ["TransactionDT", "day", "uid", "D1", "D15"],
}
SERVING = {
    "registered_model_name": "fraud-detection",
    "alias": "champion",
    "threshold_metric": "best_f1_threshold",
    "promotion_metric": "pr_auc",
}


def _raw_tables(n: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    transaction = pd.DataFrame(
        {
            KEY: np.arange(n),
            TARGET: (rng.random(n) < 0.2).astype(int),
            "TransactionDT": 86_400 + np.arange(n) * 3_600,
            "TransactionAmt": rng.uniform(1, 500, n).round(2),
            "card1": rng.choice([1000, 1001, 1002, 1003, 1004, 1005], n),
            "card4": rng.choice(["visa", "mastercard", None], n),
            "addr1": rng.choice([315.0, 325.0, np.nan], n),
            "P_emaildomain": rng.choice(["gmail.com", "mail.yahoo.com", None], n),
            "D1": rng.integers(0, 30, n).astype(float),
            "D15": rng.integers(0, 300, n).astype(float),
            "V1": rng.normal(size=n),
            "V2": rng.normal(size=n),
            "V3": rng.normal(size=n),
        }
    )
    with_identity = transaction[KEY].sample(frac=0.3, random_state=0).sort_values()
    identity = pd.DataFrame(
        {
            KEY: with_identity.to_numpy(),
            "id-01": rng.choice([-5.0, 0.0, np.nan], len(with_identity)),
            "id-02": rng.uniform(0, 1e5, len(with_identity)),
            "DeviceType": rng.choice(["mobile", "desktop"], len(with_identity)),
        }
    )
    return transaction, identity


@pytest.fixture(scope="module")
def fitted():
    """Run the batch chain on synthetic raw tables; return (bundle, raw rows, model input)."""
    transaction, identity = _raw_tables()
    merged = merge_transaction_identity(
        type_transaction(transaction, INGESTION, TARGET),
        type_identity(normalize_identity_columns(identity), INGESTION),
        INGESTION,
    )
    column_filter = fit_column_filter(merged, CLEANING, TARGET)
    stateless = add_stateless_features(apply_column_filter(merged, column_filter), FEATURES)
    encoders = fit_feature_encoders(stateless, FEATURES, TARGET)
    model_input = build_model_input(
        apply_feature_encoders(stateless, encoders), encoders, FEATURES, TARGET
    )
    options = {"model_type": "lightgbm", "device": "cpu", "n_estimators": 20, "log_every": 0}
    params = {"verbosity": -1, "min_child_samples": 5}
    model = FraudModel("lightgbm", params, options).fit(
        model_input.drop(columns=[KEY, TARGET]), model_input[TARGET], n_iterations=20
    )
    bundle = InferenceBundle(model, column_filter, encoders, INGESTION, FEATURES, TARGET)
    raw = transaction.drop(columns=TARGET).merge(identity, on=KEY, how="left")
    return bundle, raw, model_input


def _api_records(raw: pd.DataFrame) -> list[dict]:
    """Rows as the API builds them: nulls left out, codes and amounts as strings."""
    records = []
    for row in raw.astype(object).to_dict("records"):
        record = {k: v for k, v in row.items() if not pd.isna(v)}
        for col in ("card1", "TransactionAmt"):
            record[col] = str(record[col])
        records.append(record)
    return records


def test_model_input_matches_batch(fitted):
    bundle, raw, expected = fitted
    got = bundle.model_input(pd.DataFrame.from_records(_api_records(raw)))
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True),
        expected.drop(columns=TARGET).reset_index(drop=True),
        check_categorical=True,
        check_dtype=False,
    )


def test_single_rows_match_batch_scores(fitted):
    bundle, raw, expected = fitted
    batch = bundle.model.predict_proba(expected.drop(columns=[KEY, TARGET]))
    single = [
        bundle.predict_proba(pd.DataFrame.from_records([r]))[0] for r in _api_records(raw)[:40]
    ]
    np.testing.assert_allclose(single, batch[:40], rtol=1e-6)


def test_minimal_row_scores(fitted):
    bundle, _, _ = fitted
    row = {KEY: 1, "TransactionDT": 90_000, "TransactionAmt": 10.0}
    proba = bundle.predict_proba(pd.DataFrame.from_records([row]))
    assert 0 <= proba[0] <= 1


@pytest.mark.parametrize(
    "row",
    [{"TransactionDT": 90_000}, {KEY: 1, "TransactionDT": 90_000, "TransactionAmt": "abc"}],
)
def test_bad_rows_raise_value_or_key_error(fitted, row):
    """The API maps ValueError / KeyError to 422."""
    bundle, _, _ = fitted
    with pytest.raises((ValueError, KeyError)):
        bundle.predict_proba(pd.DataFrame.from_records([row]))


def _as_codes(model_input: pd.DataFrame, model: FraudModel) -> np.ndarray:
    """Pandas model input as the float matrix LightGBM sees: categories -> codes, NaN if missing."""
    X = model_input[model.feature_columns].copy()
    for col in model.categorical_columns:
        codes = X[col].cat.codes
        X[col] = codes.where(codes >= 0)
    return X.to_numpy(dtype=np.float64)


def test_fast_features_equal_pandas_features(fitted):
    bundle, raw, _ = fitted
    records = _api_records(raw)
    expected = _as_codes(bundle.model_input(pd.DataFrame.from_records(records)), bundle.model)
    np.testing.assert_array_equal(bundle._scorer.feature_matrix(records), expected)


def test_fast_scores_equal_pandas_scores(fitted):
    bundle, raw, _ = fitted
    records = _api_records(raw)
    np.testing.assert_array_equal(
        bundle.predict_records(records),
        bundle.predict_proba(pd.DataFrame.from_records(records)),
    )


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_fast_path_gives_same_scores_for_any_batch_size(fitted, batch_size):
    bundle, raw, _ = fitted
    records = _api_records(raw)
    whole = bundle.predict_records(records)
    chunked = np.concatenate(
        [
            bundle.predict_records(records[i : i + batch_size])
            for i in range(0, len(records), batch_size)
        ]
    )
    np.testing.assert_array_equal(chunked, whole)


def test_fast_path_keys_numbers_like_training(fitted):
    """card1 as 1001, 1001.0 or "1001" is the same category, as in the training data."""
    bundle, _, _ = fitted
    base = {KEY: 1, "TransactionDT": 90_000, "TransactionAmt": 10.0, "addr1": 315}
    scores = [bundle.predict_records([{**base, "card1": v}])[0] for v in (1001, 1001.0, "1001")]
    assert scores[0] == scores[1] == scores[2]


@pytest.mark.parametrize(
    "row",
    [{"TransactionDT": 90_000}, {KEY: 1, "TransactionDT": 90_000, "TransactionAmt": "abc"}],
)
def test_fast_path_rejects_bad_rows_like_pandas(fitted, row):
    bundle, _, _ = fitted
    with pytest.raises((ValueError, KeyError)):
        bundle.predict_records([row])


def test_fast_path_rejects_duplicate_ids(fitted):
    bundle, _, _ = fitted
    row = {KEY: 1, "TransactionDT": 90_000, "TransactionAmt": 10.0}
    with pytest.raises(ValueError, match="unique"):
        bundle.predict_records([row, row])


@pytest.mark.parametrize(
    ("model_type", "n_features"),
    [("lightgbm", 6), ("xgboost", None), ("hist_gb", None), ("random_forest", None)],
)
def test_fast_path_matches_pandas_for_other_models(fitted, model_type, n_features):
    """Models on a feature subset, and libraries that get a DataFrame instead of a matrix."""
    bundle, raw, model_input = fitted
    features = bundle.model.feature_columns[:n_features]
    params = {
        "lightgbm": {"verbosity": -1, "min_child_samples": 5},
        "xgboost": {"max_depth": 3},
        "hist_gb": {"max_depth": 3},
        "random_forest": {"n_estimators": 10, "random_state": 0},
    }[model_type]
    options = {"device": "cpu", "n_estimators": 10, "log_every": 0}
    model = FraudModel(model_type, params, options).fit(
        model_input[features], model_input[TARGET], n_iterations=10
    )
    other = InferenceBundle(
        model, bundle.column_filter, bundle.feature_encoders, INGESTION, FEATURES, TARGET
    )
    records = _api_records(raw)
    np.testing.assert_array_equal(
        other.predict_records(records), other.predict_proba(pd.DataFrame.from_records(records))
    )


def test_pyfunc_loads_bundle(fitted, tmp_path):
    bundle, raw, _ = fitted
    bundle.predict_records(_api_records(raw.head(2)))  # fills the caches, which are not pickled
    path = tmp_path / "bundle.pkl"
    path.write_bytes(pickle.dumps(bundle))
    pyfunc = FraudPyfunc()
    pyfunc.load_context(SimpleNamespace(artifacts={"bundle": str(path)}))
    assert not any(k.startswith("_") for k in vars(pyfunc.bundle))
    np.testing.assert_allclose(pyfunc.predict(None, raw.head(5)), bundle.predict_proba(raw.head(5)))
    records = _api_records(raw.head(5))
    np.testing.assert_array_equal(
        pyfunc.predict(None, records), bundle.predict_proba(pd.DataFrame.from_records(records))
    )


def _register(bundle, pr_auc: float) -> dict:
    evaluation = {"best_f1_threshold": 0.3, "pr_auc": pr_auc}
    with mlflow.start_run():
        return register_model(
            bundle.model,
            evaluation,
            bundle.column_filter,
            bundle.feature_encoders,
            INGESTION,
            FEATURES,
            TARGET,
            SERVING,
        )


def test_register_without_run_is_a_no_op(fitted):
    bundle, _, _ = fitted
    summary = register_model(
        bundle.model,
        {"best_f1_threshold": 0.3, "pr_auc": 0.5},
        bundle.column_filter,
        bundle.feature_encoders,
        INGESTION,
        FEATURES,
        TARGET,
        SERVING,
    )
    assert summary["version"] is None and not summary["promoted"]


def test_register_promotes_only_better_models(fitted, tmp_path, monkeypatch):
    bundle, raw, _ = fitted
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        first = _register(bundle, pr_auc=0.5)
        worse = _register(bundle, pr_auc=0.4)
        better = _register(bundle, pr_auc=0.6)
        assert (first["promoted"], worse["promoted"], better["promoted"]) == (True, False, True)

        loaded = mlflow.pyfunc.load_model("models:/fraud-detection@champion")
        assert loaded.metadata.metadata["threshold"] == 0.3
        assert loaded.metadata.metadata["pr_auc"] == 0.6
        np.testing.assert_allclose(loaded.predict(raw.head(3)), bundle.predict_proba(raw.head(3)))
        records = _api_records(raw.head(3))  # what the API sends: MLflow passes the list through
        np.testing.assert_array_equal(loaded.predict(records), bundle.predict_records(records))
    finally:
        mlflow.set_tracking_uri(None)
