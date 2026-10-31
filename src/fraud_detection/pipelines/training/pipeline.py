"""training pipeline -- not implemented yet.

Planned nodes:
- time-based split on TransactionDT (no random shuffling: fraud patterns drift)
- LightGBM with Optuna hyperparameter search, optimising PR-AUC
- log params, metrics and model to MLflow; register as challenger
"""

from kedro.pipeline import Pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([])
