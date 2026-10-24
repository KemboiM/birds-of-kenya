from kedro.pipeline import Pipeline, node

from .nodes import apply_decision, decide


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(decide, ["evaluation_report", "evaluation_per_label", "data_checks_summary", "params:version",
                      "params:paths", "params:promotion", "params:evaluation", "params:mlflow"],
             "promotion_decision", name="decide_promotion"),
        node(apply_decision, ["promotion_decision", "mlflow_run", "params:mlflow"], "production_model",
             name="apply_promotion"),
    ])
