"""Project pipelines."""

from kedro.pipeline import Pipeline

from fraud_detection.pipelines import (
    data_ingestion,
    data_validation,
    evaluation,
    feature_engineering,
    training,
)


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_validation": data_validation.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "training": training.create_pipeline(),
        "evaluation": evaluation.create_pipeline(),
    }
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    return pipelines
