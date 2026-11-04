import numpy as np
import pandas as pd
import pytest
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from fraud_detection.pipeline_registry import production_candidates, register_pipelines
from fraud_detection.pipelines.production.nodes import (
    choose_candidate,
    evaluate_holdout,
    production_options,
    select_features,
    split_holdout,
    subset_columns,
)
from fraud_detection.pipelines.production.pipeline import create_pipeline
from fraud_detection.pipelines.training.model import FraudModel

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
CANDIDATES = {
    "lgb_2": {"model": "lightgbm", "n_features": 2},
    "lgb_1": {"model": "lightgbm", "n_features": 1, "param_overrides": {"num_leaves": 7}},
}
SELECTION = {"max_latency_p99_ms": None, "pr_auc_tolerance": 1.0}  # always the fewest features
PRODUCTION = {
    "candidates": CANDIDATES,
    "selection": SELECTION,
    "holdout_fraction": 0.5,
    "param_overrides": {},
    "latency_rows": 5,
}
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
    assert select_features(IMPORTANCE, {"n_features": 2}) == ["signal", "helper"]
    everything = ["signal", "helper", "TransactionAmt", "noise"]
    assert select_features(IMPORTANCE, {"n_features": None}) == everything
    assert select_features(IMPORTANCE, {}) == everything


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
        {"num_leaves": 63, "learning_rate": 0.05, "max_depth": 8},
        {"param_overrides": {"num_leaves": 31, "max_depth": 6}},
        {"n_features": 50, "param_overrides": {"max_depth": 4}},  # the candidate's win
    )
    assert params == {"num_leaves": 31, "learning_rate": 0.05, "max_depth": 4}
    assert options["run_name"] == "production-lightgbm-50" and options["stage"] == "production"
    assert production_options(MODEL_OPTIONS, {}, {}, {})[0]["run_name"] == "production-lightgbm-all"
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


def _ev(pr_auc: float, n_features: int, p99: float) -> dict:
    return {
        "n_features": n_features,
        "holdout_pr_auc": pr_auc,
        "holdout_roc_auc": 0.9,
        "holdout_cost_per_1k": 3000.0,
        "latency_p50_ms": p99 / 2,
        "latency_p99_ms": p99,
    }


RULE_CANDIDATES = {
    "big": {"model": "xgboost", "n_features": 200},
    "mid": {"model": "lightgbm", "n_features": 100},
    "small": {"model": "lightgbm", "n_features": 50},
}


def test_choose_candidate_prefers_fewer_features_within_tolerance():
    evaluations = {"big": _ev(0.600, 200, 5), "mid": _ev(0.598, 100, 5), "small": _ev(0.590, 50, 5)}
    selection = {"max_latency_p99_ms": 20, "pr_auc_tolerance": 0.005}
    choice, report = choose_candidate(RULE_CANDIDATES, selection, **evaluations)
    # small is 0.010 behind, outside the tolerance; mid is within it and smaller than big
    assert choice["candidate"] == "mid" and choice["best_candidate"] == "big"
    assert choice["n_features"] == 100 and choice["any_fast_enough"]
    assert "| mid | lightgbm | 100 | 0.5980 |" in report and "**yes**" in report


def test_choose_candidate_drops_the_slow_ones():
    evaluations = {
        "big": _ev(0.650, 200, 40),
        "mid": _ev(0.598, 100, 15),
        "small": _ev(0.590, 50, 5),
    }
    selection = {"max_latency_p99_ms": 20, "pr_auc_tolerance": 0.005}
    choice, _ = choose_candidate(RULE_CANDIDATES, selection, **evaluations)
    assert choice["candidate"] == "mid" and choice["best_candidate"] == "mid"

    # No candidate within the budget: all of them count, and the report says so
    evaluations = {name: {**ev, "latency_p99_ms": 99.0} for name, ev in evaluations.items()}
    choice, report = choose_candidate(RULE_CANDIDATES, selection, **evaluations)
    assert choice["candidate"] == "big" and not choice["any_fast_enough"]
    assert "No candidate met the 20 ms budget" in report


def test_production_pipeline_end_to_end():
    """Two candidates, the choice, the registration summary and the copied submission."""
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
        "params:production.candidates": CANDIDATES,
        **{f"params:production.candidates.{n}": c for n, c in CANDIDATES.items()},
        "params:production.selection": SELECTION,
        "params:production.holdout_fraction": PRODUCTION["holdout_fraction"],
        "params:evaluation": EVAL_PARAMS,
        "params:serving": SERVING,
        "params:data_ingestion": {},
        "params:data_validation": {},
        "params:feature_engineering": {},
        "params:feature_engineering.join_key": KEY,
        "params:target": TARGET,
    }
    kept = (
        *(
            f"production_{n}.{d}"
            for n in CANDIDATES
            for d in ("model", "final_model", "submission")
        ),
        "production_lgb_2.evaluation",
        "production.choice",
        "production.candidates_report",
        "production.submission",
        "serving_registration",
    )
    catalog = DataCatalog(
        {
            **{name: MemoryDataset(value) for name, value in inputs.items()},
            **{name: _KeptDataset(copy_mode="assign") for name in kept},
        }
    )
    SequentialRunner().run(create_pipeline(CANDIDATES), catalog)

    for name, columns in (("lgb_2", ["signal", "helper"]), ("lgb_1", ["signal"])):
        assert catalog.load(f"production_{name}.model").feature_columns == columns
        assert catalog.load(f"production_{name}.final_model").feature_columns == columns
    assert catalog.load("production_lgb_1.model").estimator.get_params()["num_leaves"] == 7
    evaluation = catalog.load("production_lgb_2.evaluation")
    assert evaluation["holdout_rows"] == 200
    assert {"min_cost_threshold", "holdout_pr_auc", "latency_p50_ms"} <= set(evaluation)

    choice = catalog.load("production.choice")
    assert choice["candidate"] == "lgb_1"  # tolerance 1.0: the fewest features wins
    assert "| lgb_1 |" in catalog.load("production.candidates_report")
    submission = catalog.load("production.submission")
    assert submission.equals(catalog.load("production_lgb_1.submission"))
    assert len(submission) == 200 and submission[KEY].tolist() == test[KEY].tolist()
    registration = catalog.load("serving_registration")
    assert registration["candidate"] == "lgb_1" and registration["n_features"] == 1
    assert registration["version"] is None  # no MLflow run in tests


def test_registry_has_a_pipeline_per_candidate_and_a_choice():
    candidates = production_candidates()
    pipelines = register_pipelines()
    assert "serving" not in pipelines
    for name, candidate in candidates.items():
        nodes = {n.name for n in pipelines[f"production_{name}"].nodes}
        assert f"production_{name}.train_model" in nodes
        assert not any("register" in n for n in nodes)  # only the choice registers
        model = candidate["model"]
        assert {f"{model}.best_params", f"{model}.feature_importance"} <= pipelines[
            f"production_{name}"
        ].inputs()
    choose = {n.name for n in pipelines["production_choose"].nodes}
    assert {"production.choose_candidate", "production.register_model"} <= choose
    assert pipelines["production"].outputs() >= {"production.submission", "serving_registration"}


def test_production_candidates_rejects_bad_names(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "local").mkdir()
    (tmp_path / "base" / "parameters.yml").write_text(
        "production:\n  candidates:\n    xgb-200: {model: xgboost, n_features: 200}\n"
    )
    with pytest.raises(ValueError, match="xgb-200"):
        production_candidates(tmp_path)
