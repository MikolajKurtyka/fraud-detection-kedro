"""Nodes for the data_cleaning pipeline.

The column filter follows a fit / apply split: ``fit_column_filter`` learns which
columns to drop from the train split only, ``apply_column_filter`` applies that saved
decision to any data -- valid, test, or a single API request.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_time(df: pd.DataFrame, params: dict, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into an earlier train part and a later valid part on whole days.

    Random splits leak future fraud patterns into training; a time split mirrors
    production, where the model always scores transactions newer than its data.
    """
    day = df[params["time_column"]] // 86_400
    cutoff_day = int(day.quantile(params["train_fraction"]))
    is_train = day < cutoff_day
    train, valid = df[is_train], df[~is_train]

    for name, part, days in (("train", train, day[is_train]), ("valid", valid, day[~is_train])):
        logger.info(
            "Time split %s: %d rows, days %d-%d, %s rate %.4f",
            name,
            len(part),
            days.min(),
            days.max(),
            target,
            part[target].mean(),
        )
    return train, valid


def _reduce_correlated(df: pd.DataFrame, columns: list[str], threshold: float) -> dict[str, str]:
    """Within groups of identical null counts, drop one column of every highly correlated pair.

    Columns are visited from most to least unique values, so the survivor of each
    pair is the one with more unique values. Returns ``{dropped: kept_instead}``.
    """
    dropped = {}
    null_counts = df[columns].isna().sum()
    for _, group in null_counts.groupby(null_counts):
        cols = df[group.index].nunique().sort_values(ascending=False, kind="stable").index
        if len(cols) < 2:
            continue
        corr = df[cols].astype("float64").corr().abs()
        kept: list[str] = []
        for col in cols:
            twin = next((k for k in kept if corr.loc[col, k] > threshold), None)
            if twin is None:
                kept.append(col)
            else:
                dropped[col] = twin
    return dropped


def fit_column_filter(train: pd.DataFrame, params: dict, target: str) -> dict:
    """Decide which columns to drop, using the train split only.

    Rules run in order and each column is reported under the first rule that hits it.
    Protected columns (and the target) are never dropped.
    """
    protected = {*params["protected_columns"], params["join_key"], target}
    candidates = [c for c in train.columns if c not in protected]

    null_share = train[candidates].isna().mean()
    top_share = pd.Series(
        {c: train[c].value_counts(dropna=False, normalize=True).iloc[0] for c in candidates}
    )

    dropped: dict[str, str] = {}
    for col in candidates:
        if null_share[col] > params["null_threshold"]:
            dropped[col] = f"null share {null_share[col]:.4f}"
        elif top_share[col] > params["top_value_threshold"]:
            dropped[col] = f"top value share {top_share[col]:.4f}"

    v_pattern = re.compile(params["v_column_pattern"])
    v_columns = [c for c in candidates if v_pattern.match(c) and c not in dropped]
    correlated = _reduce_correlated(train, v_columns, params["v_correlation_threshold"])
    for col, twin in correlated.items():
        dropped[col] = f"|corr| > {params['v_correlation_threshold']} with {twin}"

    kept = [c for c in train.columns if c not in dropped]
    n_v_kept = sum(bool(v_pattern.match(c)) for c in kept)
    logger.info(
        "Column filter: dropping %d of %d columns (%d null/top-value, %d correlated V), "
        "keeping %d (%d V columns)",
        len(dropped),
        train.shape[1],
        len(dropped) - len(correlated),
        len(correlated),
        len(kept),
        n_v_kept,
    )
    return {
        "dropped": dropped,
        "kept": kept,
        "target": target,
        "params": {k: v for k, v in params.items() if k != "protected_columns"},
    }


def apply_column_filter(df: pd.DataFrame, column_filter: dict) -> pd.DataFrame:
    """Keep the columns chosen at fit time. Works for any number of rows, including one.

    The target may be absent (test split, API requests); any other missing column
    means the input does not match what the filter was fitted on.
    """
    kept = column_filter["kept"]
    missing = set(kept) - set(df.columns) - {column_filter["target"]}
    if missing:
        raise ValueError(f"Input is missing columns the filter expects: {sorted(missing)}")

    out = df[[c for c in kept if c in df.columns]]
    logger.info("Applied column filter: %d rows x %d columns", len(out), out.shape[1])
    return out
