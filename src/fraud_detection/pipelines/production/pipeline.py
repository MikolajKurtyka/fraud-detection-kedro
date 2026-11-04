"""production: research outputs + 05_model_input -> the Kaggle submission and the API model.

The research pipelines (training, evaluation) compare every model on all features. This one
takes one model type with its tuned params (``{model}.best_params``) and its top-N features
(``{model}.feature_importance``), and builds the single model used for both outputs:

1. Cut valid by time into valid (early stopping, threshold) and a holdout nothing is fitted
   or picked on; holdout metrics are the honest ones and decide the promotion.
2. Train on train with the chosen features, evaluate on valid, report on the holdout.
3. Refit on all labelled data -> submission_production.csv for Kaggle.
4. Register the train-only model (its threshold was picked for it) -> @champion if better.

Datasets live under the ``production`` namespace (production.model, production.evaluation,
...). The model type is chosen at pipeline level: ``production`` uses LightGBM,
``production_<model>`` the others.
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_detection.pipelines.evaluation.nodes import evaluate_model, predict_submission
from fraud_detection.pipelines.serving.nodes import register_model
from fraud_detection.pipelines.training.nodes import refit_model, train_model

from .nodes import (
    evaluate_holdout,
    production_options,
    select_features,
    split_holdout,
    subset_columns,
)

_COMMON = ["params:target", "params:feature_engineering.join_key"]


def _production_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                select_features,
                inputs=["research_importance", "params:production.n_features"],
                outputs="features",
                name="select_features",
            ),
            node(
                production_options,
                inputs=["params:model_options", "research_params", "params:production"],
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
            node(
                register_model,
                inputs=[
                    "model",
                    "evaluation",
                    "column_filter",
                    "feature_encoders",
                    "params:data_ingestion",
                    "params:feature_engineering",
                    "params:target",
                    "params:serving",
                ],
                outputs="serving_registration",
                name="register_model",
            ),
        ]
    )


def create_pipeline(model_type: str = "lightgbm", **kwargs) -> Pipeline:
    return pipeline(
        _production_pipeline(),
        namespace="production",
        inputs={
            "mdl_in_train": "mdl_in_train",
            "mdl_in_valid": "mdl_in_valid",
            "mdl_in_test": "mdl_in_test",
            "column_filter": "column_filter",
            "feature_encoders": "feature_encoders",
            "research_importance": f"{model_type}.feature_importance",
            "research_params": f"{model_type}.best_params",
        },
        outputs={"serving_registration": "serving_registration"},
        parameters={
            "params:model_options": f"params:{model_type}.model_options",
            "params:production": "params:production",
            "params:production.n_features": "params:production.n_features",
            "params:production.holdout_fraction": "params:production.holdout_fraction",
            "params:evaluation": "params:evaluation",
            "params:serving": "params:serving",
            "params:data_ingestion": "params:data_ingestion",
            "params:feature_engineering": "params:feature_engineering",
            "params:feature_engineering.join_key": "params:feature_engineering.join_key",
            "params:target": "params:target",
        },
        tags=["production", model_type],
    )
