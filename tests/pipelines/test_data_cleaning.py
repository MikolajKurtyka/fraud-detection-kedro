import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipelines.data_cleaning.nodes import (
    apply_column_filter,
    fit_column_filter,
    split_by_time,
)

TARGET = "isFraud"
PARAMS = {
    "join_key": "TransactionID",
    "time_column": "TransactionDT",
    "train_fraction": 0.8,
    "null_threshold": 0.9,
    "top_value_threshold": 0.9,
    "v_column_pattern": r"^V\d+$",
    "v_correlation_threshold": 0.95,
    "protected_columns": ["TransactionDT", "card1"],
}


@pytest.fixture
def df():
    n = 100
    rng = np.random.default_rng(0)
    v1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": np.arange(n) * 86_400,  # one row per day
            "isFraud": [1] * 3 + [0] * (n - 3),  # 97% zeros, must survive anyway
            "card1": [None] * 95 + [1.0] * 5,  # mostly null, but protected
            "mostly_null": [None] * 95 + [1.0] * 5,
            "near_constant": [0] * 95 + [1] * 5,
            "informative": rng.normal(size=n),
            "V1": v1,  # V1 / V2 / V3 share a null pattern; V2 is a copy of V1
            "V2": v1 * 2,
            "V3": rng.normal(size=n),
        }
    )


def test_split_by_time_keeps_order(df):
    train, valid = split_by_time(df, PARAMS, TARGET)
    assert len(train) + len(valid) == len(df)
    assert train["TransactionDT"].max() < valid["TransactionDT"].min()


def test_fit_column_filter_drops_by_rule(df):
    dropped = fit_column_filter(df, PARAMS, TARGET)["dropped"]
    assert "mostly_null" in dropped
    assert "near_constant" in dropped
    assert "informative" not in dropped
    assert "V3" not in dropped
    assert len({"V1", "V2"} & set(dropped)) == 1  # one of the correlated pair survives


def test_fit_column_filter_never_drops_target_or_protected(df):
    column_filter = fit_column_filter(df, PARAMS, TARGET)
    for col in ("isFraud", "TransactionID", "card1"):
        assert col in column_filter["kept"]


def test_apply_column_filter_single_row_without_target(df):
    column_filter = fit_column_filter(df, PARAMS, TARGET)
    api_row = df.drop(columns=TARGET).iloc[[0]]
    out = apply_column_filter(api_row, column_filter)
    assert len(out) == 1
    assert list(out.columns) == [c for c in column_filter["kept"] if c != TARGET]


def test_apply_column_filter_rejects_missing_columns(df):
    column_filter = fit_column_filter(df, PARAMS, TARGET)
    with pytest.raises(ValueError, match="informative"):
        apply_column_filter(df.drop(columns="informative"), column_filter)
