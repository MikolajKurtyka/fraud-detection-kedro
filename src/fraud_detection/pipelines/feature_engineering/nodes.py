"""Nodes for the feature_engineering pipeline.

Three kinds of steps:
- ``add_stateless_features``: row-by-row transforms, no fitting.
- ``fit_feature_encoders``: learns lookup tables from the train split only.
- ``apply_feature_encoders`` / ``build_model_input``: map any data through the saved
  tables -- train, valid, test, or a single API request -- with identical results.
"""

import logging

import numpy as np
import pandas as pd

from fraud_detection.keys import to_key

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400


def add_stateless_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Time, normalized D, amount, email and pseudo client ID features."""
    out = df.copy()
    seconds = out[params["time_column"]]
    out["day"] = seconds // _SECONDS_PER_DAY
    out["hour"] = ((seconds // 3600) % 24).astype("int8")
    out["weekday"] = (out["day"] % 7).astype("int8")

    amount = out["TransactionAmt"]
    out["TransactionAmt_log"] = np.log1p(amount).astype("float32")
    out["TransactionAmt_cents"] = (amount - np.floor(amount)).round(3).astype("float32")

    for col in params["d_columns"]:
        out[f"{col}_norm"] = (out["day"] - out[col]).astype("float32")

    for col in params["email_columns"]:
        parts = to_key(out[col]).str.split(".", n=2, expand=True).reindex(columns=range(3))
        for i in range(3):
            out[f"{col}_{i + 1}"] = parts[i].astype("string")

    keys = [to_key(out[c]).fillna("nan") for c in params["uid_columns"]]
    out["uid"] = keys[0].str.cat(keys[1:], sep="_")
    return out


def _feature_name(spec: dict, stat: str) -> str:
    joiner = "_to_" if spec["ratio"] else "_"
    return f"{spec['value']}{joiner}{stat}_{spec['by']}"


def fit_feature_encoders(train: pd.DataFrame, params: dict, target: str) -> dict:
    """Learn group statistics, frequencies and category encodings from train only."""
    min_size = params["min_group_size"]
    group_stats = []
    for spec in params["group_features"]:
        keys = to_key(train[spec["by"]])
        grouped = train[spec["value"]].groupby(keys).agg(["count", *spec["stats"]])
        grouped = grouped[grouped["count"] >= min_size]
        for stat in spec["stats"]:
            group_stats.append(
                {
                    **spec,
                    "stat": stat,
                    "name": _feature_name(spec, stat),
                    "mapping": grouped[stat].dropna().to_dict(),
                }
            )

    frequency = {
        col: to_key(train[col]).value_counts(normalize=True).to_dict()
        for col in params["frequency_columns"]
    }

    excluded = {params["join_key"], target, *params["drop_columns"]}
    categorical = {}
    for col in train.columns:
        if col in excluded or pd.api.types.is_numeric_dtype(train[col]):
            continue
        values = sorted(to_key(train[col]).dropna().unique())
        if len(values) <= params["max_native_categories"]:
            categorical[col] = {"encoding": "native", "categories": values}
        else:
            categorical[col] = {"encoding": "code", "codes": {v: i for i, v in enumerate(values)}}

    encoders = {"group_stats": group_stats, "frequency": frequency, "categorical": categorical}
    transformed = apply_feature_encoders(train.head(1), encoders)
    encoders["feature_columns"] = [c for c in transformed.columns if c not in excluded]

    n_native = sum(c["encoding"] == "native" for c in categorical.values())
    logger.info(
        "Fitted encoders: %d group features, %d frequency columns, "
        "%d native categorical + %d coded categorical, %d model features",
        len(group_stats),
        len(frequency),
        n_native,
        len(categorical) - n_native,
        len(encoders["feature_columns"]),
    )
    return encoders


def apply_feature_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Map data through the fitted tables. Unseen groups give NaN, unseen values 0 frequency."""
    new = {}
    for spec in encoders["group_stats"]:
        stat = to_key(df[spec["by"]]).map(spec["mapping"]).astype("float64")
        if spec["ratio"]:
            stat = (df[spec["value"]] / stat).replace([np.inf, -np.inf], np.nan)
        new[spec["name"]] = stat.astype("float32")

    for col, mapping in encoders["frequency"].items():
        keys = to_key(df[col])
        new[f"{col}_freq"] = keys.map(mapping).astype("float64").fillna(0).where(keys.notna())
        new[f"{col}_freq"] = new[f"{col}_freq"].astype("float32")

    out = df.copy()
    for col, enc in encoders["categorical"].items():
        keys = to_key(out[col]).astype(object)
        if enc["encoding"] == "native":
            known = keys.where(keys.isin(enc["categories"]))  # unseen -> NaN
            out[col] = pd.Categorical(known, categories=enc["categories"])
        else:
            out[col] = keys.map(enc["codes"]).astype("float32")

    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def build_model_input(df: pd.DataFrame, encoders: dict, params: dict, target: str) -> pd.DataFrame:
    """Keep the ID, the target (if present) and the fitted feature columns, in fixed order."""
    features = encoders["feature_columns"]
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    leading = [params["join_key"]] + ([target] if target in df.columns else [])
    out = df[leading + features]
    logger.info("Model input: %d rows x %d features", len(out), len(features))
    return out
