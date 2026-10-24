from kedro.pipeline import Pipeline, node

from .nodes import train


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(train, ["data_checks_summary", "params:version", "params:paths", "params:training"],
             "train_config", name="train_with_main_py"),
    ])
