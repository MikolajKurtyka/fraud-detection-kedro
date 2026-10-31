import numpy as np
import pandas as pd
import pytest

from fraud_detection.pipelines.feature_engineering.nodes import (
    add_stateless_features,
    apply_feature_encoders,
    build_model_input,
    fit_feature_encoders,
)

TARGET = "isFraud"
PARAMS = {
    "join_key": "TransactionID",
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


@pytest.fixture
def df():
    n = 40
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "isFraud": rng.integers(0, 2, n).astype("int8"),
            "TransactionDT": 86_400 + np.arange(n) * 7_200,
            "TransactionAmt": rng.uniform(1, 500, n).round(2),
            "card1": pd.Categorical([str(1000 + i % 6) for i in range(n)]),  # 6 values -> coded
            "card4": pd.Categorical(["visa", "mastercard"] * (n // 2)),  # 2 values -> native
            "addr1": pd.Categorical(["315", None, "325", "315"] * (n // 4)),
            "D1": rng.integers(0, 30, n).astype("float32"),
            "D15": rng.integers(0, 300, n).astype("float32"),
            "P_emaildomain": pd.Categorical(
                ["gmail.com", "mail.yahoo.com", None, "aol.com"] * (n // 4)
            ),
        }
    )


@pytest.fixture
def fitted(df):
    stateless = add_stateless_features(df, PARAMS)
    return stateless, fit_feature_encoders(stateless, PARAMS, TARGET)


def test_stateless_features(df):
    out = add_stateless_features(df, PARAMS)
    assert out["day"].iloc[0] == 1
    assert out["D1_norm"].iloc[0] == 1 - df["D1"].iloc[0]
    assert out.loc[1, ["P_emaildomain_1", "P_emaildomain_2", "P_emaildomain_3"]].tolist() == [
        "mail",
        "yahoo",
        "com",
    ]
    assert out["uid"].iloc[0].startswith("1000_315_")


def test_encoders_choose_native_or_code(fitted):
    _, encoders = fitted
    assert encoders["categorical"]["card4"]["encoding"] == "native"
    assert encoders["categorical"]["card1"]["encoding"] == "code"
    assert "uid" not in encoders["categorical"]
    assert "TransactionAmt_to_mean_card1" in encoders["feature_columns"]
    assert "TransactionDT" not in encoders["feature_columns"]


def test_single_row_matches_batch(df, fitted):
    """The API scores one row at a time; its features must equal the batch result."""
    _, encoders = fitted
    batch = build_model_input(
        apply_feature_encoders(add_stateless_features(df, PARAMS), encoders),
        encoders,
        PARAMS,
        TARGET,
    )
    for i in (0, 1, 7):
        row = df.drop(columns=TARGET).iloc[[i]]
        single = build_model_input(
            apply_feature_encoders(add_stateless_features(row, PARAMS), encoders),
            encoders,
            PARAMS,
            TARGET,
        )
        expected = batch.drop(columns=TARGET).iloc[[i]]
        pd.testing.assert_frame_equal(single, expected)


def test_unseen_values(df, fitted):
    _, encoders = fitted
    row = df.iloc[[0]].copy()
    row["card1"] = pd.Categorical(["99999"])
    row["card4"] = pd.Categorical(["unknown-network"])
    out = apply_feature_encoders(add_stateless_features(row, PARAMS), encoders)
    assert np.isnan(out["card1"].iloc[0])  # unseen code
    assert pd.isna(out["card4"].iloc[0])  # unseen native category
    assert out["card1_freq"].iloc[0] == 0  # never seen in train
    assert np.isnan(out["TransactionAmt_to_mean_card1"].iloc[0])  # no group stats
