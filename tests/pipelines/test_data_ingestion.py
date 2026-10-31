import pandas as pd
import pytest

from fraud_detection.pipelines.data_ingestion.nodes import (
    expand_columns,
    merge_transaction_identity,
    normalize_identity_columns,
    type_identity,
    type_transaction,
)

TARGET = "isFraud"
PARAMS = {
    "join_key": "TransactionID",
    "schema": {
        "transaction": {"categorical": ["card1", "addr1"], "numerical": ["TransactionAmt"]},
        "identity": {"categorical": ["DeviceType"], "numerical": ["id_{01..01}"]},
    },
    "numerical_dtype": "float32",
    "numerical_dtype_overrides": {"TransactionAmt": "float64"},
}


@pytest.fixture
def transaction():
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [10.0, 25.5, 99.9],
            "card1": [13926, 2755, 4663],
            "addr1": [315.0, None, 315.0],
        }
    )


@pytest.fixture
def identity():
    return pd.DataFrame({"TransactionID": [2], "id-01": [-5.0], "DeviceType": ["mobile"]})


def test_expand_columns():
    assert expand_columns(["C{1..3}", "ProductCD"]) == ["C1", "C2", "C3", "ProductCD"]
    assert expand_columns(["id_{09..11}"]) == ["id_09", "id_10", "id_11"]


def test_normalize_identity_columns(identity):
    assert "id_01" in normalize_identity_columns(identity).columns


def test_type_transaction_casts_dtypes(transaction):
    out = type_transaction(transaction, PARAMS, TARGET)
    assert out["TransactionID"].dtype == "int64"
    assert out[TARGET].dtype == "int8"
    assert out["TransactionAmt"].dtype == "float64"
    assert out["card1"].dtype == "category"
    # Float-stored codes become clean strings, nulls stay null
    assert out["addr1"].tolist()[0] == "315"
    assert out["addr1"].isna().tolist() == [False, True, False]


def test_type_transaction_works_without_target(transaction):
    out = type_transaction(transaction.drop(columns=TARGET), PARAMS, TARGET)
    assert TARGET not in out.columns


def test_type_transaction_rejects_schema_drift(transaction):
    with pytest.raises(ValueError, match="Unexpected"):
        type_transaction(transaction.assign(extra=1), PARAMS, TARGET)
    with pytest.raises(ValueError, match="Missing"):
        type_transaction(transaction.drop(columns="card1"), PARAMS, TARGET)


def test_type_transaction_rejects_bad_target(transaction):
    with pytest.raises(ValueError, match="0/1"):
        type_transaction(transaction.assign(isFraud=[0, 2, 0]), PARAMS, TARGET)


def test_type_identity(identity):
    out = type_identity(normalize_identity_columns(identity), PARAMS)
    assert out["id_01"].dtype == "float32"
    assert out["DeviceType"].dtype == "category"


def test_merge_keeps_all_transactions(transaction, identity):
    merged = merge_transaction_identity(
        type_transaction(transaction, PARAMS, TARGET),
        type_identity(normalize_identity_columns(identity), PARAMS),
        PARAMS,
    )
    assert len(merged) == len(transaction)
    assert merged["has_identity"].tolist() == [0, 1, 0]
    assert merged.loc[merged.TransactionID == 2, "DeviceType"].item() == "mobile"
    assert merged["DeviceType"].dtype == "category"


def test_merge_rejects_duplicate_identity(transaction, identity):
    with pytest.raises(ValueError):
        merge_transaction_identity(transaction, pd.concat([identity, identity]), PARAMS)
