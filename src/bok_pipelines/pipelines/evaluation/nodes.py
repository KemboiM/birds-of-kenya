"""Evaluation: a standard report for a trained version, logged to MLflow, with the model
registered in the MLflow model registry under the alias "candidate"."""

import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd

from ... import REPO_ROOT
from ...project import (as_version, label_list_path, model_path, read_json, read_label_list, read_split,
                        train_counts)
from ...reliability.evaluation import evaluation_report, report_from_saved
from ...reliability.schemas import PER_LABEL

log = logging.getLogger(__name__)
PREDICTION_COLUMNS = ["file", "label", "predicted", "confidence", "correct", "top3"]


def _counts(paths, version):
    try:
        return train_counts(paths, version)
    except FileNotFoundError:
        return None


def _predict(version, paths, e, counts):
    """Run the saved model over this version's test photos (the same crops main.py tested on)."""
    import keras
    from src.data.loader import image_dataset

    labels = read_label_list(paths, version)
    index = {c: i for i, c in enumerate(labels)}
    test = [(f, label, box) for f, label, box in read_split(paths, version, "test") if label in index]
    files = [os.path.join(paths["data_root"], f) for f, _, _ in test]
    boxes = [box or [0.0, 0.0, 1.0, 1.0] for _, _, box in test] if any(b for _, _, b in test) else None
    model = keras.models.load_model(model_path(paths, version))
    size = model.input_shape[1]
    log.info("Predicting %d test photos with %s", len(files), model_path(paths, version))
    probs = model.predict(image_dataset(files, (size, size), boxes=boxes, batch_size=e["batch_size"]), verbose=0)
    y = np.array([index[label] for _, label, _ in test])

    summary, per_label = evaluation_report(y, probs, labels, counts, e["rare_max_train"], e["confidence_cutoff"],
                                           e["top_k"], e["calibration_bins"])
    top = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    predictions = pd.DataFrame({
        "file": [f for f, _, _ in test], "label": [label for _, label, _ in test],
        "predicted": [labels[t[0]] for t in top], "confidence": probs.max(axis=1).round(4),
        "correct": [labels[t[0]] == label for t, (_, label, _) in zip(top, test)],
        "top3": ["; ".join(f"{labels[j]} {probs[i, j]:.2f}" for j in t) for i, t in enumerate(top)],
    })
    return summary, per_label, predictions


def evaluate(train_config, version, paths, e):
    version = as_version(version)
    source = e["source"]
    if source == "auto":
        source = "predict" if os.path.exists(model_path(paths, version)) else "reports"
    counts = _counts(paths, version)
    saved = read_json(os.path.join("reports", version, "metrics.json"))

    if source == "predict":
        summary, per_label, predictions = _predict(version, paths, e, counts)
        if saved:  # same model, same test photos: the result must match what main.py reported
            gap = abs(summary["accuracy"] - saved["accuracy"])
            summary["accuracy_in_training_report"] = saved["accuracy"]
            summary["matches_training_report"] = bool(gap <= 0.005)
            if gap > 0.005:
                log.warning("Accuracy %.4f differs from reports/%s/metrics.json (%.4f)",
                            summary["accuracy"], version, saved["accuracy"])
    else:
        per_species = os.path.join("reports", version, "per_species_2x2.csv")
        if not saved or not os.path.exists(per_species):
            raise FileNotFoundError(f"No model file and no reports/{version}/metrics.json and per_species_2x2.csv "
                                    f"to evaluate version {version} from.")
        summary, per_label = report_from_saved(saved, pd.read_csv(per_species), counts, e["rare_max_train"])
        summary.pop("misclassified_images", None)
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)

    summary = {"version": version, "source": "model re-run on the test photos" if source == "predict"
               else f"reports/{version} written by main.py", "trained_on": train_config.get("created"), **summary}
    for k in ("accuracy", "top5_accuracy", "f1_macro", "expected_calibration_error",
              "accuracy_rare_labels", "accuracy_common_labels"):
        if summary.get(k) is not None:
            log.info("%-28s %.4f", k, summary[k])
    return summary, PER_LABEL.validate(per_label), predictions


def _scalar(v):
    return v is None or isinstance(v, (str, int, float, bool))


def _register_model(client, version, paths, m):
    """Register the .keras model once for each version; later runs reuse that registered version."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    name = m["registered_model"]
    found = [mv for mv in client.search_model_versions(f"name='{name}'")
             if mv.tags.get("birds_version") == version]
    if found:
        return max(int(mv.version) for mv in found)
    signature = ModelSignature(  # in: image paths; out: one row for each image
        inputs=Schema([ColSpec("string", "path")]),
        outputs=Schema([ColSpec("string", "path"), ColSpec("string", "label"),
                        ColSpec("double", "confidence"), ColSpec("string", "top3")]))
    with warnings.catch_warnings():  # an input example would have to be a photo path on this computer
        warnings.filterwarnings("ignore", message=".*input example was not provided.*")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=os.path.join(REPO_ROOT, "src", "bok_pipelines", "reliability", "mlflow_model.py"),
            artifacts={"keras_model": model_path(paths, version), "labels": label_list_path(paths, version)},
            registered_model_name=name,
            signature=signature,
            pip_requirements=["tensorflow>=2.16", "pillow>=10.0", "pandas", "numpy", "mlflow>=3.0"],
        )
    mv = int(info.registered_model_version)
    client.set_model_version_tag(name, str(mv), "birds_version", version)
    return mv


def track(report, per_label, train_config, version, paths, m):
    """Log the run to MLflow: training settings, metrics, data checks, reports and the model."""
    version = as_version(version)
    if not m.get("enabled"):
        return {"enabled": False, "version": version}
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment"])
    client = MlflowClient()
    data_checks = read_json(os.path.join("reports", version, "data_checks.json"))
    with mlflow.start_run(run_name=f"evaluate {version}") as run:
        mlflow.set_tags({"birds_version": version, "stage": "evaluation", "evaluation_source": report["source"]})
        mlflow.log_params({k: v for k, v in train_config.items() if _scalar(v)})
        mlflow.log_metrics({k: float(v) for k, v in report.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_dict(report, "evaluation_report.json")
        mlflow.log_table(per_label, "evaluation_per_species.json")
        if data_checks:
            mlflow.log_metric("data_checks_passed", float(data_checks["passed"]))
            mlflow.log_dict(data_checks, "data_checks.json")
        for f in sorted(glob.glob(os.path.join("reports", version, "*"))):
            if os.path.isfile(f):
                mlflow.log_artifact(f, "reports")
        mv = None
        if m.get("log_model") and os.path.exists(model_path(paths, version)):
            mv = _register_model(client, version, paths, m)
            client.set_registered_model_alias(m["registered_model"], "candidate", str(mv))
            mlflow.set_tag("registered_model_version", mv)
            log.info("Registered %s version %s (alias: candidate)", m["registered_model"], mv)
    return {"enabled": True, "version": version, "tracking_uri": m["tracking_uri"], "experiment": m["experiment"],
            "run_id": run.info.run_id, "registered_model": m["registered_model"], "model_version": mv}
