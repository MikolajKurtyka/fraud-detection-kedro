import pandas as pd
import pytest

from fraud_detection.pipelines.data_ingestion.nodes import (
    merge_transaction_identity,
    normalize_identity_columns,
    reduce_memory,
)

PARAMS = {"join_key": "TransactionID", "downcast_numerics": True}


@pytest.fixture
def transaction():
    return pd.DataFrame(
        {"TransactionID": [1, 2, 3], "TransactionAmt": [10.0, 25.5, 99.9], "isFraud": [0, 1, 0]}
    )


@pytest.fixture
def identity():
    return pd.DataFrame({"TransactionID": [2], "id-01": [-5.0], "DeviceType": ["mobile"]})


def test_normalize_identity_columns(identity):
    assert "id_01" in normalize_identity_columns(identity).columns


def test_merge_keeps_all_transactions(transaction, identity):
    merged = merge_transaction_identity(transaction, normalize_identity_columns(identity), PARAMS)
    assert len(merged) == len(transaction)
    assert merged["has_identity"].tolist() == [0, 1, 0]
    assert merged.loc[merged.TransactionID == 2, "DeviceType"].item() == "mobile"


def test_merge_rejects_duplicate_identity(transaction, identity):
    with pytest.raises(ValueError):
        merge_transaction_identity(transaction, pd.concat([identity, identity]), PARAMS)


def test_reduce_memory_downcasts(transaction):
    out = reduce_memory(transaction, PARAMS)
    assert out["TransactionAmt"].dtype == "float32"
    assert out["isFraud"].dtype == "int8"
    pd.testing.assert_frame_equal(out.astype("float64"), transaction.astype("float64"))
