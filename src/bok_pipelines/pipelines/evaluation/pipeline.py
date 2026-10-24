from kedro.pipeline import Pipeline, node

from .nodes import evaluate, track


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(evaluate, ["train_config", "params:version", "params:paths", "params:evaluation"],
             ["evaluation_report", "evaluation_per_label", "test_predictions"], name="evaluate_version"),
        node(track, ["evaluation_report", "evaluation_per_label", "train_config", "params:version",
                     "params:paths", "params:mlflow"],
             "mlflow_run", name="log_to_mlflow"),
    ])
