"""Reliability components for machine-learning pipelines.

    data_checks  - data-validation checks before training
    evaluation   - standard evaluation report before a model is used
    gate         - promotion gate: replace the production model only when the new one is better
    monitoring   - drift checks on new data and a human review queue
    mlflow_model - MLflow wrapper for a saved Keras image classifier

Nothing here is specific to birds; the bird project wires these into Kedro pipelines
in bok_pipelines.pipelines.
"""
