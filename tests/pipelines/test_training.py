import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipelines.training import model as model_module
from fraud_detection.pipelines.training.model import MODEL_TYPES, FraudModel, undersample
from fraud_detection.pipelines.training.nodes import (
    cross_validate,
    refit_model,
    train_model,
    tune_hyperparameters,
)

TARGET = "isFraud"
KEY = "TransactionID"

# CPU everywhere so tests run without a GPU
OPTIONS = {
    "lightgbm": {"device": "cpu", "params": {"verbosity": -1, "min_child_samples": 5}},
    "xgboost": {"device": "cpu", "params": {"tree_method": "hist"}},
    "catboost": {"device": "CPU", "params": {"depth": 4}},
    "logreg": {"params": {"max_iter": 200}, "one_hot_min_frequency": 5},
    "random_forest": {"params": {"n_estimators": 20, "min_samples_leaf": 5, "random_state": 0}},
    "hist_gb": {"params": {"min_samples_leaf": 5}},
}


def _options(model_type: str, **overrides) -> dict:
    return {
        "model_type": model_type,
        "n_estimators": 30,
        "early_stopping_rounds": 10,
        "log_every": 0,
        "tuning_metric": "pr_auc",
        "n_trials": 0,
        "timeout_minutes": 1,
        "search_space": {"learning_rate": {"type": "float", "low": 0.05, "high": 0.3}},
        "refit": True,
        "refit_iteration_multiplier": 1.2,
        **OPTIONS[model_type],
        **overrides,
    }


def _frame(n: int, seed: int, start: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            KEY: range(start, start + n),
            TARGET: (signal + rng.normal(scale=0.5, size=n) > 1.2).astype("int8"),
            "signal": signal.astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
            "card4": pd.Categorical(
                rng.choice(["visa", "mastercard", None], n), categories=["mastercard", "visa"]
            ),
        }
    )


@pytest.fixture
def data():
    return _frame(600, 0), _frame(300, 1, start=600)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_train_model_and_score_single_row(data, model_type):
    train, valid = data
    options = _options(model_type)
    model, metrics, importance = train_model(train, valid, options["params"], options, TARGET, KEY)

    assert metrics["valid_roc_auc"] > 0.8
    assert set(importance["feature"]) == {"signal", "noise", "card4"}
    assert importance["feature"].iloc[0] == "signal"
    # API path: one row, no target, no ID column
    single = model.predict_proba(valid.drop(columns=[KEY, TARGET]).iloc[[0]])
    assert single.shape == (1,) and 0 <= single[0] <= 1


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_class_weight(data, model_type):
    train, valid = data
    options = _options(model_type, imbalance={"method": "class_weight"})
    model, metrics, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    assert metrics["valid_roc_auc"] > 0.7  # weighted LightGBM overfits a bit on 600 rows
    if model_type == "xgboost":
        y = train[TARGET]
        expected = (y == 0).sum() / (y == 1).sum()
        assert model.estimator.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_undersample_keeps_all_positives():
    df = _frame(1000, 0)
    X, y = undersample(df, df[TARGET], ratio=2)
    n_pos = int(df[TARGET].sum())
    assert int(y.sum()) == n_pos
    assert (y == 0).sum() == 2 * n_pos
    assert X.index.is_monotonic_increasing  # original (time) order kept


def test_undersample_only_touches_train(data, monkeypatch):
    calls = []

    def spy(X, y, ratio, seed=42):
        calls.append(len(X))
        return undersample(X, y, ratio, seed)

    monkeypatch.setattr(model_module, "undersample", spy)
    train, valid = data
    options = _options("lightgbm", imbalance={"method": "undersample", "undersample_ratio": 2})
    _, metrics, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    assert calls == [len(train)]  # called once, on train; valid is never resampled
    assert metrics["valid_roc_auc"] > 0.8


def test_unknown_imbalance_method():
    with pytest.raises(ValueError):
        FraudModel("lightgbm", {}, {"imbalance": {"method": "smote"}})


@pytest.mark.parametrize("model_type", ["lightgbm", "xgboost", "catboost", "hist_gb"])
def test_refit_uses_scaled_iterations(data, model_type):
    train, valid = data
    options = _options(model_type)
    model, _, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    refitted = refit_model(train, valid, model, options["params"], options, TARGET, KEY)

    assert refitted is not model
    assert refitted.best_iteration == max(1, round(model.best_iteration * 1.2))
    single = refitted.predict_proba(valid.drop(columns=[KEY, TARGET]).iloc[[0]])
    assert 0 <= single[0] <= 1


def test_refit_non_iterative_model(data):
    train, valid = data
    options = _options("random_forest")
    model, _, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    refitted = refit_model(train, valid, model, options["params"], options, TARGET, KEY)
    assert refitted is not model and refitted.best_iteration is None


def test_refit_off_returns_train_only_model(data):
    train, valid = data
    options = _options("lightgbm", refit=False)
    model, _, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    assert refit_model(train, valid, model, options["params"], options, TARGET, KEY) is model


def test_fit_without_valid_needs_iterations(data):
    train, _ = data
    model = FraudModel("lightgbm", {"verbosity": -1}, _options("lightgbm"))
    with pytest.raises(ValueError):
        model.fit(train.drop(columns=[KEY, TARGET]), train[TARGET])


def test_cross_validate_expanding_folds(data):
    train, valid = data
    options = _options("lightgbm", cv_folds=3)
    model, _, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    result = cross_validate(train, valid, model, options["params"], options, TARGET, KEY)

    assert result["cv_folds"] == 3 and len(result["folds"]) == 3
    train_rows = [f["train_rows"] for f in result["folds"]]
    assert train_rows == sorted(train_rows) and train_rows[0] < train_rows[-1]
    assert all(f["valid_rows"] == (len(train) + len(valid)) // 4 for f in result["folds"])
    assert 0.5 < result["cv_roc_auc_mean"] <= 1 and result["cv_roc_auc_std"] >= 0


def test_cross_validate_off(data):
    train, valid = data
    options = _options("lightgbm")
    model, _, _ = train_model(train, valid, options["params"], options, TARGET, KEY)
    assert cross_validate(train, valid, model, options["params"], options, TARGET, KEY) == {
        "cv_folds": 0
    }


def test_tune_hyperparameters_returns_merged_params(data):
    train, valid = data
    options = _options("lightgbm", n_trials=2)
    params = tune_hyperparameters(train, valid, options, TARGET, KEY)
    assert params["verbosity"] == -1  # base param kept
    assert 0.05 <= params["learning_rate"] <= 0.3  # tuned param added


def test_tune_skipped_when_no_trials(data):
    train, valid = data
    options = _options("lightgbm")
    assert tune_hyperparameters(train, valid, options, TARGET, KEY) == options["params"]


def test_unknown_model_type():
    with pytest.raises(ValueError):
        FraudModel("svm", {}, {})
