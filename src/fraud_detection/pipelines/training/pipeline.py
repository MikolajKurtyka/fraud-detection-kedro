"""training: 05_model_input -> 06_models + 08_reporting, one namespaced pipeline per model.

Per model: Optuna tuning (optional) -> final fit with early stopping on valid ->
time-series CV (optional) -> refit on train + valid for the submission (optional).
Datasets are namespaced: lightgbm.model, xgboost.metrics, logreg.final_model, ...
"""

from kedro.pipeline import Pipeline, node, pipeline

from .model import MODEL_TYPES
from .nodes import cross_validate, refit_model, train_model, tune_hyperparameters

_COMMON_INPUTS = ["params:target", "params:feature_engineering.join_key"]


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                tune_hyperparameters,
                inputs=["mdl_in_train", "mdl_in_valid", "params:model_options", *_COMMON_INPUTS],
                outputs="best_params",
                name="tune_hyperparameters",
            ),
            node(
                train_model,
                inputs=[
                    "mdl_in_train",
                    "mdl_in_valid",
                    "best_params",
                    "params:model_options",
                    *_COMMON_INPUTS,
                ],
                outputs=["model", "metrics", "feature_importance"],
                name="train_model",
            ),
            node(
                cross_validate,
                inputs=[
                    "mdl_in_train",
                    "mdl_in_valid",
                    "model",
                    "best_params",
                    "params:model_options",
                    *_COMMON_INPUTS,
                ],
                outputs="cv_metrics",
                name="cross_validate",
            ),
            node(
                refit_model,
                inputs=[
                    "mdl_in_train",
                    "mdl_in_valid",
                    "model",
                    "best_params",
                    "params:model_options",
                    *_COMMON_INPUTS,
                ],
                outputs="final_model",
                name="refit_model",
            ),
        ]
    )


def create_pipeline(models: tuple[str, ...] = MODEL_TYPES, **kwargs) -> Pipeline:
    return sum(
        (
            pipeline(
                _model_pipeline(),
                namespace=model,
                inputs={"mdl_in_train", "mdl_in_valid"},
                parameters=set(_COMMON_INPUTS),
                tags=["training", model],
            )
            for model in models
        ),
        Pipeline([]),
    )
