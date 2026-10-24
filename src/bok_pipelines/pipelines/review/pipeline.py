from kedro.pipeline import Pipeline, node

from .nodes import build_review_queue, collect_field_predictions, monitoring_report


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(collect_field_predictions, "params:paths", "field_predictions", name="collect_field_predictions"),
        node(build_review_queue, ["field_predictions", "params:paths", "params:review"], "review_queue",
             name="build_review_queue"),
        node(monitoring_report, ["field_predictions", "review_queue", "image_index", "params:version",
                                 "params:review", "params:mlflow"],
             "monitoring_report", name="monitoring_report"),
    ])
