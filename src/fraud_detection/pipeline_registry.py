"""Project pipelines."""

from kedro.pipeline import Pipeline

from fraud_detection.pipelines import (
    data_cleaning,
    data_ingestion,
    data_validation,
    evaluation,
    feature_engineering,
    serving,
    training,
)
from fraud_detection.pipelines.training.model import MODEL_TYPES


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_validation": data_validation.create_pipeline(),
        "data_cleaning": data_cleaning.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "training": training.create_pipeline(),
        "evaluation": evaluation.create_pipeline(),
        "serving": serving.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    # Single-model shortcuts (not part of __default__, they duplicate "training" nodes)
    for model in MODEL_TYPES:
        pipelines[f"training_{model}"] = training.create_pipeline(models=(model,))
        pipelines[f"serving_{model}"] = serving.create_pipeline(model_type=model)
    return pipelines
