import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipelines.evaluation.nodes import (
    blend_submissions,
    evaluate_model,
    predict_submission,
)
from fraud_detection.pipelines.evaluation.pipeline import create_pipeline
from fraud_detection.pipelines.training.model import BOOSTING_TYPES, MODEL_TYPES, FraudModel

TARGET = "isFraud"
KEY = "TransactionID"
PARAMS = {"fixed_threshold": 0.5, "target_fpr": 0.01}


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            KEY: range(n),
            TARGET: (signal + rng.normal(scale=0.5, size=n) > 1.2).astype("int8"),
            "signal": signal.astype("float32"),
            "noise": rng.normal(size=n).astype("float32"),
        }
    )


@pytest.fixture
def model_and_valid():
    train, valid = _frame(600, 0), _frame(300, 1)
    options = {"n_estimators": 30, "early_stopping_rounds": 10, "log_every": 0, "device": "cpu"}
    model = FraudModel("lightgbm", {"verbosity": -1, "min_child_samples": 5}, options)
    model.fit(
        train.drop(columns=[KEY, TARGET]),
        train[TARGET],
        valid.drop(columns=[KEY, TARGET]),
        valid[TARGET],
    )
    return model, valid


def test_evaluate_model_metrics(model_and_valid):
    model, valid = model_and_valid
    metrics, figure = evaluate_model(model, valid, PARAMS, TARGET, KEY)
    for prefix in ("fixed", "best_f1", "at_fpr"):
        assert 0 <= metrics[f"{prefix}_precision"] <= 1
        assert 0 <= metrics[f"{prefix}_recall"] <= 1
        counts = sum(metrics[f"{prefix}_{c}"] for c in ("tp", "fp", "fn", "tn"))
        assert counts == len(valid)
    assert metrics["at_fpr_fpr"] <= PARAMS["target_fpr"] + 1e-9
    assert metrics["best_f1_f1"] >= metrics["fixed_f1"] - 1e-9  # best F1 is the max over thresholds
    assert figure is not None


def test_predict_submission_format(model_and_valid):
    model, valid = model_and_valid
    test = valid.drop(columns=TARGET)
    submission = predict_submission(model, test, KEY)
    assert list(submission.columns) == [KEY, "isFraud"]
    assert len(submission) == len(test)
    assert submission["isFraud"].between(0, 1).all()


def test_blend_rank_average():
    a = pd.DataFrame({KEY: [1, 2, 3], "isFraud": [0.1, 0.5, 0.9]})
    b = pd.DataFrame({KEY: [1, 2, 3], "isFraud": [0.2, 0.3, 0.99]})
    blend = blend_submissions(a, b)
    assert blend["isFraud"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_blend_rejects_mismatched_ids():
    a = pd.DataFrame({KEY: [1, 2], "isFraud": [0.1, 0.5]})
    b = pd.DataFrame({KEY: [2, 1], "isFraud": [0.1, 0.5]})
    with pytest.raises(ValueError):
        blend_submissions(a, b)


def test_pipeline_blends_boosting_models_and_submits_final_model():
    pipe = create_pipeline(models=MODEL_TYPES)
    blend = next(n for n in pipe.nodes if n.name == "blend_submissions")
    assert sorted(blend.inputs) == sorted(f"{m}.submission" for m in BOOSTING_TYPES)
    submit = next(n for n in pipe.nodes if n.name == "lightgbm.predict_submission")
    assert "lightgbm.final_model" in submit.inputs
