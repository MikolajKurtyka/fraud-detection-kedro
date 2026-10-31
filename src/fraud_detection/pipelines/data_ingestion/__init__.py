"""Load raw Kaggle CSVs, join transaction + identity tables, store compact parquet."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
