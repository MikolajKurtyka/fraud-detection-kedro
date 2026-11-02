import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipelines.data_validation.nodes import DataValidationError, validate_data

TARGET = "isFraud"
INGESTION = {
    "join_key": "TransactionID",
    "schema": {
        "transaction": {
            "categorical": ["ProductCD", "card1"],
            "numerical": ["TransactionDT", "TransactionAmt", "C{1..2}"],
        },
        "identity": {"categorical": ["DeviceType"], "numerical": ["id_{01..01}"]},
    },
    "numerical_dtype": "float32",
    "numerical_dtype_overrides": {"TransactionDT": "int64", "TransactionAmt": "float64"},
}
PARAMS = {
    "time_column": "TransactionDT",
    "required_columns": ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"],
    "value_ranges": {"TransactionDT": {"ge": 0}, "TransactionAmt": {"gt": 0}},
    "fraud_rate_range": [0.01, 0.5],
    "max_null_rate_diff": 0.25,
    "fail_on_error": True,
}


def _table(n: int, start: int, seed: int, labelled: bool) -> pd.DataFrame:
    """A table shaped like int_{split}: typed columns, sorted by time, IDs unique."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(start, start + n, dtype="int64"),
            "TransactionDT": np.arange(start * 100, (start + n) * 100, 100, dtype="int64"),
            "TransactionAmt": rng.uniform(1, 500, n),
            "ProductCD": pd.Categorical(rng.choice(["W", "C"], n)),
            "card1": pd.Categorical(rng.choice(["1000", "2000", None], n)),
            "C1": rng.normal(size=n).astype("float32"),
            "C2": np.where(rng.random(n) < 0.2, np.nan, 1.0).astype("float32"),
            "DeviceType": pd.Categorical(rng.choice(["mobile", None], n)),
            "id_01": rng.normal(size=n).astype("float32"),
            "has_identity": rng.integers(0, 2, n).astype("int8"),
        }
    )
    if labelled:
        df.insert(1, TARGET, (rng.random(n) < 0.1).astype("int8"))
    return df


@pytest.fixture
def train():
    return _table(200, 0, 0, labelled=True)


@pytest.fixture
def test():
    return _table(100, 1000, 1, labelled=False)


def _failed_checks(train, test, params=PARAMS) -> str:
    with pytest.raises(DataValidationError) as err:
        validate_data(train, test, INGESTION, params, TARGET)
    return str(err.value)


def test_clean_data_passes(train, test):
    report = validate_data(train, test, INGESTION, PARAMS, TARGET)
    assert report["passed"]
    assert report["splits"]["train"]["rows"] == 200
    assert report["parity_failures"] == []


def test_schema_failures_are_all_reported(train, test):
    train.loc[3, "TransactionID"] = train.loc[2, "TransactionID"]
    train.loc[5, "TransactionAmt"] = -1.0
    test["ProductCD"] = test["ProductCD"].where(test.index != 0)  # one null, still a category
    message = _failed_checks(train, test)
    assert "train: TransactionID field_uniqueness" in message
    assert "train: TransactionAmt greater_than(0)" in message
    assert "test: ProductCD not_nullable" in message


def test_wrong_dtype_and_extra_column(train, test):
    train["C1"] = train["C1"].astype("float64")
    test["unexpected"] = 1
    message = _failed_checks(train, test)
    assert "train: C1 dtype('float32')" in message
    assert "column_in_schema" in message
    assert "test_has_no_extra_columns" in message


def test_time_order_and_test_after_train(train, test):
    train = train.iloc[::-1].reset_index(drop=True)
    test = test.assign(TransactionDT=test["TransactionDT"] - 200_000)
    message = _failed_checks(train, test)
    assert "TransactionDT_sorted" in message
    assert "test_after_train" in message


def test_fraud_rate_out_of_range(train, test):
    message = _failed_checks(train.assign(isFraud=np.int8(0)), test)
    assert "fraud_rate_in_range" in message


def test_null_rate_drift(train, test):
    test["C1"] = np.float32(np.nan)
    message = _failed_checks(train, test)
    assert "null_rate_diff" in message
    assert "'C1': 1.0" in message


def test_warn_only_returns_failed_report(train, test):
    train.loc[5, "TransactionAmt"] = 0.0
    report = validate_data(train, test, INGESTION, {**PARAMS, "fail_on_error": False}, TARGET)
    assert not report["passed"]
    (failure,) = report["splits"]["train"]["failures"]
    assert failure["column"] == "TransactionAmt"
    assert failure["n_failures"] == 1
