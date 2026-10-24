"""Pipelines that `kedro run --pipeline <name>` can run."""

from kedro.pipeline import Pipeline

from .pipelines import data_checks, evaluation, promotion, review, training


def register_pipelines() -> dict[str, Pipeline]:
    p = {
        "data_checks": data_checks.create_pipeline(),
        "training": training.create_pipeline(),
        "evaluation": evaluation.create_pipeline(),
        "promotion": promotion.create_pipeline(),
        "review": review.create_pipeline(),
    }
    # A new version end to end: check the data, train, evaluate, then promote only if it is better.
    p["release"] = p["data_checks"] + p["training"] + p["evaluation"] + p["promotion"]
    # Default (kedro run): check, evaluate and promote a version that is already trained.
    p["__default__"] = p["data_checks"] + p["evaluation"] + p["promotion"]
    return p
