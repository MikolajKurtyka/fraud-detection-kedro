"""production: research outputs + 05_model_input -> the Kaggle submission and the API model.

The research pipelines (training, evaluation) compare every model on all features. This one
builds a few **candidates** (``production.candidates``: a model type with its research tuned
params, ``{model}.best_params``, on its top-N research features, ``{model}.feature_importance``)
and chooses one of them for both outputs.

Per candidate (namespace ``production_<name>``, e.g. production_xgboost_200.pkl):

1. Cut valid by time into valid (early stopping, threshold) and a holdout nothing is fitted
   or picked on.
2. Train on train with the chosen features, evaluate on valid, report on the holdout.
3. Refit on all labelled data -> submission_production_<name>.csv.

Then ``choose_candidate`` applies ``production.selection`` to the holdout reports (latency
budget, best PR-AUC, fewest features within a tolerance), registers the chosen train-only
model (@champion moves only if it beats the champion) and copies its submission to
submission_production.csv.
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_detection.pipelines.evaluation.nodes import evaluate_model, predict_submission
from fraud_detection.pipelines.training.nodes import refit_model, train_model

from .nodes import (
    choose_candidate,
    copy_submission,
    evaluate_holdout,
    production_options,
    register_chosen,
    select_features,
    split_holdout,
    subset_columns,
)

_COMMON = ["params:target", "params:feature_engineering.join_key"]
_SHARED_PARAMS = [
    "params:production",
    "params:production.holdout_fraction",
    "params:evaluation",
    "params:serving",
    "params:feature_engineering.join_key",
    "params:target",
]


def _candidate_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                select_features,
                inputs=["research_importance", "params:candidate"],
                outputs="features",
                name="select_features",
            ),
            node(
                production_options,
                inputs=[
                    "params:model_options",
                    "research_params",
                    "params:production",
                    "params:candidate",
                ],
                outputs=["options", "best_params"],
                name="production_options",
            ),
            *(
                node(
                    subset_columns,
                    inputs=[f"mdl_in_{split}", "features", *_COMMON],
                    outputs=f"{split}_selected",
                    name=f"subset_columns_{split}",
                )
                for split in ("train", "valid")
            ),
            node(
                split_holdout,
                inputs=[
                    "mdl_in_valid",
                    "params:production.holdout_fraction",
                    "params:feature_engineering.join_key",
                ],
                outputs=["prod_valid", "prod_holdout"],
                name="split_holdout",
            ),
            node(
                train_model,
                inputs=["train_selected", "prod_valid", "best_params", "options", *_COMMON],
                outputs=["model", "metrics", "feature_importance"],
                name="train_model",
            ),
            node(
                evaluate_model,
                inputs=["model", "prod_valid", "params:evaluation", *_COMMON],
                outputs=["valid_evaluation", "evaluation_curves"],
                name="evaluate_model",
            ),
            node(
                evaluate_holdout,
                inputs=[
                    "model",
                    "prod_holdout",
                    "valid_evaluation",
                    "params:evaluation",
                    "params:production",
                    "params:serving",
                    *_COMMON,
                ],
                outputs="evaluation",
                name="evaluate_holdout",
            ),
            node(
                refit_model,
                inputs=[
                    "train_selected",
                    "valid_selected",
                    "model",
                    "best_params",
                    "options",
                    *_COMMON,
                ],
                outputs="final_model",
                name="refit_model",
            ),
            node(
                predict_submission,
                inputs=["final_model", "mdl_in_test", "params:feature_engineering.join_key"],
                outputs="submission",
                name="predict_submission",
            ),
        ]
    )


def candidate_pipeline(name: str, candidate: dict) -> Pipeline:
    """One candidate under the ``production_<name>`` namespace; registers nothing."""
    model_type = candidate["model"]
    return pipeline(
        _candidate_pipeline(),
        namespace=f"production_{name}",
        inputs={
            "mdl_in_train": "mdl_in_train",
            "mdl_in_valid": "mdl_in_valid",
            "mdl_in_test": "mdl_in_test",
            "research_importance": f"{model_type}.feature_importance",
            "research_params": f"{model_type}.best_params",
        },
        parameters={
            "params:model_options": f"params:{model_type}.model_options",
            "params:candidate": f"params:production.candidates.{name}",
            **{p: p for p in _SHARED_PARAMS},
        },
        tags=["production", model_type, f"production_{name}"],
    )


def choose_pipeline(candidates: dict) -> Pipeline:
    """Choose among the candidates' saved outputs, register the winner, copy its submission."""
    names = list(candidates)
    return Pipeline(
        [
            node(
                choose_candidate,
                inputs={
                    "candidates": "params:production.candidates",
                    "selection": "params:production.selection",
                    **{n: f"production_{n}.evaluation" for n in names},
                },
                outputs=["production.choice", "production.candidates_report"],
                name="production.choose_candidate",
            ),
            node(
                register_chosen,
                inputs={
                    "choice": "production.choice",
                    "column_filter": "column_filter",
                    "feature_encoders": "feature_encoders",
                    "ingestion_params": "params:data_ingestion",
                    "feature_params": "params:feature_engineering",
                    "target": "params:target",
                    "serving_params": "params:serving",
                    "validation_params": "params:data_validation",
                    **{f"{n}__model": f"production_{n}.model" for n in names},
                    **{f"{n}__evaluation": f"production_{n}.evaluation" for n in names},
                },
                outputs="serving_registration",
                name="production.register_model",
            ),
            node(
                copy_submission,
                inputs={
                    "choice": "production.choice",
                    **{n: f"production_{n}.submission" for n in names},
                },
                outputs="production.submission",
                name="production.copy_submission",
            ),
        ],
        tags=["production", "production_choose"],
    )


def create_pipeline(candidates: dict, **kwargs) -> Pipeline:
    """Every candidate, then the choice."""
    return sum(
        (candidate_pipeline(name, c) for name, c in candidates.items()),
        choose_pipeline(candidates),
    )
