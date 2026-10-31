"""Turn column values into stable string keys for categories and lookup tables."""

import pandas as pd


def to_key(series: pd.Series) -> pd.Series:
    """Return values as strings, with nulls kept as ``<NA>``.

    Numbers are formatted value by value -- integer values without a decimal part
    (``315.0`` -> ``"315"``), others as-is -- so a single API row produces exactly the
    same keys as the full training table did.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        numbers = series.astype("float64")
        is_int = numbers.notna() & (numbers % 1 == 0)
        keys = numbers.astype("string")
        keys[is_int] = numbers[is_int].astype("Int64").astype("string")
        return keys
    return series.astype("string")
