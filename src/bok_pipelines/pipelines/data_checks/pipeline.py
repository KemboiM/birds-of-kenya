from kedro.pipeline import Pipeline, node

from .nodes import build_image_index, run_checks


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(build_image_index, ["params:paths", "params:data_checks"], "image_index",
             name="index_training_photos"),
        node(run_checks, ["image_index", "params:version", "params:paths", "params:data_checks",
                          "params:training"],
             ["data_checks_summary", "data_check_issues"], name="run_data_checks"),
    ])
