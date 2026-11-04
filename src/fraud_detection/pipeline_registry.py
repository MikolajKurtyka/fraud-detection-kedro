"""Project pipelines.

Research: data_* -> feature_engineering -> training -> evaluation (every model, all features).
Production: a few candidates (model type + top-N features) -> the chosen one is the Kaggle
submission and the API model.
"""

import re
from pathlib import Path

from kedro.pipeline import Pipeline

from fraud_detection.pipelines import (
    data_cleaning,
    data_ingestion,
    data_validation,
    evaluation,
    feature_engineering,
    production,
    training,
)
from fraud_detection.pipelines.training.model import MODEL_TYPES
from fraud_detection.settings import CONFIG_LOADER_ARGS, CONFIG_LOADER_CLASS

_CONF = Path(__file__).resolve().parents[2] / "conf"
# Used as Python keyword names and pipeline names, so: lowercase identifiers, none reserved
_CANDIDATE_NAME = re.compile(r"[a-z][a-z0-9_]*")
_RESERVED = {"choose", "candidates", "selection", "choice"}


def production_candidates(conf_source: Path = _CONF) -> dict:
    """``production.candidates`` from parameters_production.yml.

    Read when the pipelines are built, since each candidate is its own set of nodes; so
    a ``--params`` override cannot add or remove a candidate, only change one.
    """
    params = CONFIG_LOADER_CLASS(conf_source=str(conf_source), **CONFIG_LOADER_ARGS)["parameters"]
    candidates = params["production"]["candidates"]
    for name, candidate in candidates.items():
        if not _CANDIDATE_NAME.fullmatch(name) or name in _RESERVED:
            raise ValueError(f"Invalid production candidate name {name!r}")
        if candidate["model"] not in MODEL_TYPES:
            raise ValueError(f"Candidate {name!r}: unknown model {candidate['model']!r}")
    return candidates


def register_pipelines() -> dict[str, Pipeline]:
    candidates = production_candidates()
    pipelines = {
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_validation": data_validation.create_pipeline(),
        "data_cleaning": data_cleaning.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "training": training.create_pipeline(),
        "evaluation": evaluation.create_pipeline(),
        "production": production.create_pipeline(candidates),
    }
    pipelines["__default__"] = sum(pipelines.values(), Pipeline([]))
    # Shortcuts, not part of __default__ (they duplicate nodes above)
    for model in MODEL_TYPES:
        pipelines[f"training_{model}"] = training.create_pipeline(models=(model,))
    for name, candidate in candidates.items():
        pipelines[f"production_{name}"] = production.candidate_pipeline(name, candidate)
    pipelines["production_choose"] = production.choose_pipeline(candidates)
    return pipelines
