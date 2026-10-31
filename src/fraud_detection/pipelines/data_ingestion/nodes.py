"""Nodes for the data_ingestion pipeline."""

import logging
import re

import pandas as pd

from fraud_detection.keys import to_key

logger = logging.getLogger(__name__)

_RANGE_PATTERN = re.compile(r"^(?P<prefix>.*)\{(?P<start>\d+)\.\.(?P<end>\d+)\}$")


def expand_columns(specs: list[str]) -> list[str]:
    """Expand ``"C{1..3}"`` into ``["C1", "C2", "C3"]``; plain names pass through.

    A zero-padded start keeps the padding: ``"id_{01..03}"`` -> ``id_01 ... id_03``.
    """
    columns = []
    for spec in specs:
        match = _RANGE_PATTERN.match(spec)
        if match is None:
            columns.append(spec)
            continue
        start, end = match["start"], match["end"]
        width = len(start) if start.startswith("0") else 0
        columns += [f"{match['prefix']}{i:0{width}d}" for i in range(int(start), int(end) + 1)]
    return columns


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    """Align column names between train and test identity tables.

    The Kaggle test identity file uses ``id-01`` while train uses ``id_01``.
    """
    renamed = identity.rename(columns=lambda c: c.replace("-", "_"))
    n_renamed = sum(old != new for old, new in zip(identity.columns, renamed.columns, strict=True))
    if n_renamed:
        logger.info("Renamed %d identity columns ('-' -> '_')", n_renamed)
    return renamed


def _to_categorical(series: pd.Series) -> pd.Series:
    """Store codes as categories of strings.

    Numeric codes (card1, addr1, ...) are read from CSV as int/float; ``to_key``
    gives "315" instead of "315.0", so train, test and API codes match.
    """
    return to_key(series).astype("category")


def _apply_column_types(
    df: pd.DataFrame, table: str, params: dict, target: str | None = None
) -> pd.DataFrame:
    """Validate the columns of one raw table against the schema and cast their dtypes."""
    key = params["join_key"]
    schema = params["schema"][table]
    categorical = expand_columns(schema["categorical"])
    numerical = expand_columns(schema["numerical"])

    expected = {key, *categorical, *numerical}
    missing = expected - set(df.columns)
    unexpected = set(df.columns) - expected - {target}
    if missing or unexpected:
        raise ValueError(
            f"{table}: columns do not match schema. "
            f"Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
        )

    out = df.copy()

    if out[key].isna().any() or out[key].duplicated().any():
        raise ValueError(f"{table}: {key} must be non-null and unique")
    out[key] = out[key].astype("int64")

    if target is not None and target in out.columns:
        if out[target].isna().any() or not out[target].isin([0, 1]).all():
            raise ValueError(f"{table}: target {target} must contain only 0/1")
        out[target] = out[target].astype("int8")
        logger.info("%s: target %s found, positive rate %.4f", table, target, out[target].mean())
    elif target is not None:
        logger.info("%s: target %s not present (unlabelled split)", table, target)

    overrides = params["numerical_dtype_overrides"]
    for col in numerical:
        dtype = overrides.get(col, params["numerical_dtype"])
        out[col] = pd.to_numeric(out[col], errors="raise").astype(dtype)

    for col in categorical:
        out[col] = _to_categorical(out[col])

    logger.info(
        "%s: typed %d rows, %d categorical + %d numerical columns, %.0f MB -> %.0f MB",
        table,
        len(out),
        len(categorical),
        len(numerical),
        df.memory_usage(deep=True).sum() / 1e6,
        out.memory_usage(deep=True).sum() / 1e6,
    )
    return out


def type_transaction(transaction: pd.DataFrame, params: dict, target: str) -> pd.DataFrame:
    """Apply the transaction schema. The target is optional so test data passes too."""
    return _apply_column_types(transaction, "transaction", params, target=target)


def type_identity(identity: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the identity schema (no target in this table)."""
    return _apply_column_types(identity, "identity", params)


def merge_transaction_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Left-join identity onto transactions (identity exists for ~24% of rows)."""
    key = params["join_key"]
    if identity[key].duplicated().any():
        raise ValueError(f"Identity table has duplicated {key} values")

    orphans = ~identity[key].isin(transaction[key])
    if orphans.any():
        logger.warning("%d identity rows have no matching transaction", orphans.sum())

    merged = transaction.merge(identity, on=key, how="left", validate="one_to_one")
    merged["has_identity"] = merged[key].isin(identity[key]).astype("int8")
    logger.info(
        "Merged %d transactions x %d columns, %.1f%% with identity data, %.0f MB",
        len(merged),
        merged.shape[1],
        100 * merged["has_identity"].mean(),
        merged.memory_usage(deep=True).sum() / 1e6,
    )
    return merged
