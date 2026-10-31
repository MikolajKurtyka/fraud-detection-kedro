"""evaluation pipeline -- not implemented yet.

Planned nodes:
- PR-AUC, ROC-AUC, recall at fixed false-positive rate
- cost-based threshold: minimise (missed fraud x amount) + (false alarm x review cost)
- SHAP global importance + per-prediction explanations
- promote challenger to champion alias if it beats the current champion
"""

from kedro.pipeline import Pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([])
