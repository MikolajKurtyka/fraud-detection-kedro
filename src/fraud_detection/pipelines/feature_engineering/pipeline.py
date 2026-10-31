"""feature_engineering pipeline -- not implemented yet.

Planned nodes:
- time features from TransactionDT (hour, weekday, day index)
- client UID proxy from card1 + addr1 + D1-derived account start day
- frequency / aggregation features per UID (amount mean/std, counts)
- shared with serving via kedro-mlflow's pipeline_ml_factory (no training/serving skew)
"""

from kedro.pipeline import Pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([])
