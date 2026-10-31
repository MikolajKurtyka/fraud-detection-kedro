"""evaluation: 06_models + 05_model_input -> 07_model_output + 08_reporting.

Per model: precision / recall / F1 and curves on valid from the train-only model (also logged
into the model's MLflow training run), and a Kaggle submission for the test set from the final
(refitted) model. With two or more gradient-boosting models, a rank-average blend submission is
added. Baselines are for comparison only and stay out of the blend.
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_detection.pipelines.training.model import BOOSTING_TYPES, MODEL_TYPES

from .nodes import blend_submissions, evaluate_model, predict_submission


def _model_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                evaluate_model,
                inputs=[
                    "model",
                    "mdl_in_valid",
                    "params:evaluation",
                    "params:target",
                    "params:feature_engineering.join_key",
                ],
                outputs=["evaluation", "evaluation_curves"],
                name="evaluate_model",
            ),
            node(
                predict_submission,
                inputs=["final_model", "mdl_in_test", "params:feature_engineering.join_key"],
                outputs="submission",
                name="predict_submission",
            ),
        ]
    )


def create_pipeline(models: tuple[str, ...] = MODEL_TYPES, **kwargs) -> Pipeline:
    per_model = sum(
        (
            pipeline(
                _model_pipeline(),
                namespace=model,
                inputs={"mdl_in_valid", "mdl_in_test"},
                parameters={
                    "params:evaluation",
                    "params:target",
                    "params:feature_engineering.join_key",
                },
                tags=["evaluation", model],
            )
            for model in models
        ),
        Pipeline([]),
    )
    blended = [model for model in models if model in BOOSTING_TYPES]
    if len(blended) < 2:
        return per_model
    blend = node(
        blend_submissions,
        inputs=[f"{model}.submission" for model in blended],
        outputs="blend.submission",
        name="blend_submissions",
        tags=["evaluation"],
    )
    return per_model + Pipeline([blend])
