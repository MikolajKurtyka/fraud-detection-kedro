"""Nodes for the data_ingestion pipeline."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    """Align column names between train and test identity tables.

    The Kaggle test identity file uses ``id-01`` while train uses ``id_01``.
    """
    return identity.rename(columns=lambda c: c.replace("-", "_"))


def merge_transaction_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Left-join identity onto transactions (identity exists for ~24% of rows)."""
    key = params["join_key"]
    if identity[key].duplicated().any():
        raise ValueError(f"Identity table has duplicated {key} values")

    merged = transaction.merge(identity, on=key, how="left", validate="one_to_one")
    merged["has_identity"] = merged[key].isin(identity[key]).astype("int8")
    logger.info(
        "Merged %d transactions, %.1f%% with identity data",
        len(merged),
        100 * merged["has_identity"].mean(),
    )
    return merged


def reduce_memory(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Downcast numeric columns; roughly halves memory (train: ~2.1 GB -> ~1.1 GB)."""
    if not params["downcast_numerics"]:
        return df
    before = df.memory_usage(deep=True).sum()
    out = df.copy()
    for col in out.select_dtypes(include=[np.floating]).columns:
        out[col] = out[col].astype(np.float32)
    for col in out.select_dtypes(include=[np.integer]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    after = out.memory_usage(deep=True).sum()
    logger.info("Memory reduced from %.0f MB to %.0f MB", before / 1e6, after / 1e6)
    return out
