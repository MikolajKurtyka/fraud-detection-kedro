import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from fraud_detection.pipeline_registry import register_pipelines
from fraud_detection.pipelines.production.nodes import (
    evaluate_holdout,
    production_options,
    select_features,
    split_holdout,
    subset_columns,
)
from fraud_detection.pipelines.production.pipeline import create_pipeline
from fraud_detection.pipelines.training.model import MODEL_TYPES, FraudModel

TARGET = "isFraud"
KEY = "TransactionID"
COST = {"amount_column": "TransactionAmt", "review_cost": 5.0, "false_block_cost": 15.0}
EVAL_PARAMS = {"fixed_threshold": 0.5, "target_fpr": 0.01, "cost": COST}
SERVING = {
    "registered_model_name": "fraud-detection-test",
    "alias": "champion",
    "threshold_metric": "min_cost_threshold",
    "promotion_metric": "holdout_pr_auc",
}
PRODUCTION = {"n_features": 2, "holdout_fraction": 0.5, "param_overrides": {}, "latency_rows": 5}
MODEL_OPTIONS = {
    "model_type": "lightgbm",
    "device": "cpu",
    "n_estimators": 30,
    "early_stopping_rounds": 10,
    "log_every": 0,
    "refit": True,
    "refit_iteration_multiplier": 1.2,
    "params": {"verbosity": -1, "min_child_samples": 5},
}


def _frame(n: int, seed: int, start: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            KEY: range(start, start + n),
            TARGET: (signal + rng.normal(scale=0.5, size=n) > 1.2).astype("int8"),
            "signal": signal.astype("float32"),
            "helper": (signal + rng.normal(scale=2, size=n)).astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "TransactionAmt": rng.lognormal(4, 1, size=n),
        }
    )


class _KeptDataset(MemoryDataset):
    """Not released after its last consumer, so the test can read it after the run."""

    def _release(self) -> None:
        pass


IMPORTANCE = pd.DataFrame(
    {"feature": ["noise", "signal", "TransactionAmt", "helper"], "importance": [1, 9, 2, 5]}
)


def test_select_features_top_n_and_all():
    assert select_features(IMPORTANCE, 2) == ["signal", "helper"]
    assert select_features(IMPORTANCE, None) == ["signal", "helper", "TransactionAmt", "noise"]


def test_subset_columns_keeps_id_and_target():
    df = _frame(10, 0)
    assert list(subset_columns(df, ["signal"], TARGET, KEY).columns) == [KEY, TARGET, "signal"]
    with pytest.raises(KeyError, match="missing"):
        subset_columns(df, ["not_a_column"], TARGET, KEY)


def test_split_holdout_takes_the_latest_rows():
    df = _frame(100, 0).sample(frac=1, random_state=0)  # shuffled
    prod_valid, holdout = split_holdout(df, 0.3, KEY)
    assert (len(prod_valid), len(holdout)) == (70, 30)
    assert prod_valid[KEY].max() < holdout[KEY].min()


def test_production_options_override_params_and_name_the_run():
    options, params = production_options(
        MODEL_OPTIONS,
        {"num_leaves": 63, "learning_rate": 0.05},
        {"param_overrides": {"num_leaves": 31}},
    )
    assert params == {"num_leaves": 31, "learning_rate": 0.05}
    assert options["run_name"] == "production-lightgbm" and options["stage"] == "production"
    assert options["n_estimators"] == MODEL_OPTIONS["n_estimators"]


def test_evaluate_holdout_uses_the_valid_threshold():
    train, holdout = _frame(600, 0), _frame(300, 1, start=600)
    model = FraudModel("lightgbm", MODEL_OPTIONS["params"], MODEL_OPTIONS)
    model.fit(train[["signal", "helper"]], train[TARGET], n_iterations=20)
    evaluation = {"min_cost_threshold": 0.3, "pr_auc": 0.5}

    result = evaluate_holdout(
        model, holdout, evaluation, EVAL_PARAMS, PRODUCTION, SERVING, TARGET, KEY
    )

    assert result["pr_auc"] == 0.5  # the valid evaluation is kept
    assert result["holdout_threshold"] == 0.3
    assert result["holdout_rows"] == 300 and result["n_features"] == 2
    counts = sum(result[f"holdout_{c}"] for c in ("tp", "fp", "fn", "tn"))
    assert counts == 300
    assert 0.5 < result["holdout_roc_auc"] <= 1
    assert result["latency_p99_ms"] >= result["latency_p50_ms"] > 0


def test_production_pipeline_end_to_end():
    train, valid, test = _frame(600, 0), _frame(400, 1, start=600), _frame(200, 2, start=1000)
    inputs = {
        "mdl_in_train": train,
        "mdl_in_valid": valid,
        "mdl_in_test": test.drop(columns=[TARGET]),
        "column_filter": {},
        "feature_encoders": {},
        "lightgbm.feature_importance": IMPORTANCE,
        "lightgbm.best_params": MODEL_OPTIONS["params"],
        "params:lightgbm.model_options": MODEL_OPTIONS,
        "params:production": PRODUCTION,
        "params:production.n_features": PRODUCTION["n_features"],
        "params:production.holdout_fraction": PRODUCTION["holdout_fraction"],
        "params:evaluation": EVAL_PARAMS,
        "params:serving": SERVING,
        "params:data_ingestion": {},
        "params:feature_engineering": {},
        "params:feature_engineering.join_key": KEY,
        "params:target": TARGET,
    }
    kept = (
        "production.model",
        "production.final_model",
        "production.evaluation",
        "production.submission",
        "serving_registration",
    )
    catalog = DataCatalog(
        {
            **{name: MemoryDataset(value) for name, value in inputs.items()},
            **{name: _KeptDataset(copy_mode="assign") for name in kept},
        }
    )
    SequentialRunner().run(create_pipeline(), catalog)

    model, final_model = catalog.load("production.model"), catalog.load("production.final_model")
    assert model.feature_columns == ["signal", "helper"]
    assert final_model.feature_columns == ["signal", "helper"]
    evaluation = catalog.load("production.evaluation")
    assert evaluation["holdout_rows"] == 200
    assert {"min_cost_threshold", "holdout_pr_auc", "latency_p50_ms"} <= set(evaluation)
    submission = catalog.load("production.submission")
    assert len(submission) == 200 and submission[KEY].tolist() == test[KEY].tolist()
    registration = catalog.load("serving_registration")
    assert registration["model_type"] == "lightgbm" and registration["version"] is None


def test_registry_has_a_production_pipeline_per_model():
    pipelines = register_pipelines()
    assert "serving" not in pipelines
    for model in MODEL_TYPES:
        names = {n.name for n in pipelines[f"production_{model}"].nodes}
        assert "production.register_model" in names
    research_inputs = pipelines["production_xgboost"].inputs()
    assert {"xgboost.best_params", "xgboost.feature_importance"} <= research_inputs
