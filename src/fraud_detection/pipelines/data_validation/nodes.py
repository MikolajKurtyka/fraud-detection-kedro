"""Nodes for the data_validation pipeline: pandera checks on the joined, typed tables.

data_ingestion already fails on missing or unexpected columns and on a target outside {0, 1}.
These checks cover what it does not: unique IDs, value ranges, required (never null) columns,
time order, the fraud rate, and train/test parity (same columns and dtypes, test after train,
similar null rates).
"""

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from fraud_detection.pipelines.data_ingestion.nodes import expand_columns

logger = logging.getLogger(__name__)

TABLES = ("transaction", "identity")
_RANGE_CHECKS = {"gt": pa.Check.gt, "ge": pa.Check.ge, "lt": pa.Check.lt, "le": pa.Check.le}


class DataValidationError(ValueError):
    """The input data failed one or more checks."""


def build_schema(
    ingestion_params: dict, params: dict, target: str, labelled: bool
) -> pa.DataFrameSchema:
    """Schema of an ``int_{split}`` table, from the data_ingestion column types."""
    schema = ingestion_params["schema"]
    key, time = ingestion_params["join_key"], params["time_column"]
    required = set(params["required_columns"])
    overrides = ingestion_params["numerical_dtype_overrides"]

    def checks(col: str) -> list[pa.Check]:
        ranges = params["value_ranges"].get(col, {})
        return [_RANGE_CHECKS[op](bound) for op, bound in ranges.items()]

    columns = {key: pa.Column("int64", unique=True, nullable=False)}
    for table in TABLES:
        for col in expand_columns(schema[table]["numerical"]):
            dtype = overrides.get(col, ingestion_params["numerical_dtype"])
            columns[col] = pa.Column(dtype, checks(col), nullable=col not in required)
        for col in expand_columns(schema[table]["categorical"]):
            columns[col] = pa.Column("category", nullable=col not in required)
    columns["has_identity"] = pa.Column("int8", pa.Check.isin([0, 1]), nullable=False)

    frame_checks = [
        pa.Check(lambda df: df[time].is_monotonic_increasing, name=f"{time}_sorted"),
    ]
    if labelled:
        columns[target] = pa.Column("int8", pa.Check.isin([0, 1]), nullable=False)
        low, high = params["fraud_rate_range"]
        frame_checks.append(
            pa.Check(lambda df: low <= df[target].mean() <= high, name="fraud_rate_in_range")
        )
    return pa.DataFrameSchema(columns, checks=frame_checks, strict=True)


def _schema_failures(df: pd.DataFrame, schema: pa.DataFrameSchema) -> list[dict]:
    """Every failed check (lazy validation), one entry per column and check."""
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases.astype({"column": str, "check": str})
        return [
            {
                "column": column,
                "check": check,
                "n_failures": len(group),
                "examples": [str(v) for v in group["failure_case"].head(3)],
            }
            for (column, check), group in cases.groupby(["column", "check"], dropna=False)
        ]
    return []


def _parity_failures(
    train: pd.DataFrame, test: pd.DataFrame, params: dict, target: str
) -> tuple[list[dict], dict]:
    """Test has train's columns and dtypes (minus the target), starts after train, and has
    similar null rates. Returns the failures and the largest null-rate difference."""
    failures = []
    expected = [c for c in train.columns if c != target]
    if missing := sorted(set(expected) - set(test.columns)):
        failures.append({"check": "test_has_train_columns", "columns": missing})
    if extra := sorted(set(test.columns) - set(expected)):
        failures.append({"check": "test_has_no_extra_columns", "columns": extra})

    common = [c for c in expected if c in test.columns]
    kinds = [
        c
        for c in common
        if isinstance(train[c].dtype, pd.CategoricalDtype)
        != isinstance(test[c].dtype, pd.CategoricalDtype)
        or (not isinstance(train[c].dtype, pd.CategoricalDtype) and train[c].dtype != test[c].dtype)
    ]
    if kinds:
        failures.append({"check": "same_dtypes", "columns": kinds})

    time = params["time_column"]
    if test[time].min() <= train[time].max():
        failures.append(
            {
                "check": "test_after_train",
                "train_max": int(train[time].max()),
                "test_min": int(test[time].min()),
            }
        )

    diff = (test[common].isna().mean() - train[common].isna().mean()).abs()
    drifted = diff[diff > params["max_null_rate_diff"]].sort_values(ascending=False)
    if not drifted.empty:
        failures.append(
            {
                "check": "null_rate_diff",
                "max_allowed": params["max_null_rate_diff"],
                "columns": {c: round(float(v), 4) for c, v in drifted.items()},
            }
        )
    null_rate = {"max_diff": round(float(diff.max()), 4), "column": str(diff.idxmax())}
    return failures, null_rate


def validate_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ingestion_params: dict,
    params: dict,
    target: str,
) -> dict:
    """Run every check on int_train / int_test; raise if any fails (``fail_on_error``)."""
    splits = {}
    for split, df, labelled in (("train", train, True), ("test", test, False)):
        schema = build_schema(ingestion_params, params, target, labelled)
        splits[split] = {
            "rows": len(df),
            "columns": df.shape[1],
            "failures": _schema_failures(df, schema),
        }
    parity, null_rate = _parity_failures(train, test, params, target)
    report = {
        "passed": not parity and not any(s["failures"] for s in splits.values()),
        "fraud_rate": round(float(train[target].mean()), 5),
        "null_rate_train_vs_test": null_rate,
        "splits": splits,
        "parity_failures": parity,
    }

    failed = [
        f"{split}: {f['column']} {f['check']} ({f['n_failures']} failures, e.g. {f['examples']})"
        for split, s in splits.items()
        for f in s["failures"]
    ] + [f"parity: {f}" for f in parity]
    if not failed:
        logger.info(
            "Data validation passed: train %d rows, test %d rows, fraud rate %.4f, "
            "largest train/test null-rate difference %.3f (%s)",
            len(train),
            len(test),
            report["fraud_rate"],
            null_rate["max_diff"],
            null_rate["column"],
        )
        return report
    message = "Data validation failed:\n  " + "\n  ".join(failed)
    if params["fail_on_error"]:
        raise DataValidationError(message)
    logger.warning(message)
    return report
