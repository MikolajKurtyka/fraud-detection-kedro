"""serving: 06_models + fitted artifacts -> MLflow registry (one pyfunc model for the API).

Packages the column filter, feature encoders and one model type into an ``InferenceBundle``
(raw rows in, fraud probability out), registers it, and moves ``@champion`` to it when it
beats the current champion on valid. The model type is chosen at pipeline level:
``serving`` uses LightGBM, ``serving_<model>`` the others.
"""

from kedro.pipeline import Pipeline, node

from .nodes import register_model


def create_pipeline(model_type: str = "lightgbm", **kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                register_model,
                inputs=[
                    f"{model_type}.model",
                    f"{model_type}.evaluation",
                    "column_filter",
                    "feature_encoders",
                    "params:data_ingestion",
                    "params:feature_engineering",
                    "params:target",
                    "params:serving",
                ],
                outputs="serving_registration",
                name=f"register_{model_type}",
            )
        ],
        tags=["serving", model_type],
    )
