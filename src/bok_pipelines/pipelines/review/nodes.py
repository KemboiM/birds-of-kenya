"""Review and monitoring of photos identified in the field with identify.py.

1. Collect every bird from outputs/identify/*_identified.json into one table.
2. Queue the uncertain birds for a person to check. Labels already entered in the queue are kept.
3. Report how the field photos and the model's confidence compare with training and testing (drift),
   and how many reviewed birds the model got right.
"""

import glob
import json
import logging
import os

import pandas as pd

from ...project import as_version, read_json
from ...reliability.data_checks import index_images
from ...reliability.monitoring import drift_report, psi, review_queue
from ...reliability.schemas import FIELD_PREDICTIONS

log = logging.getLogger(__name__)
COLUMNS = ["photo", "bird_id", "predicted", "identified", "confidence", "detector_score", "box", "top3",
           "result_file"]
REVIEW_COLUMNS = ["correct_label", "reviewer", "reviewed_on"]


def collect_field_predictions(paths):
    rows = []
    for f in sorted(glob.glob(os.path.join(paths["identify_outputs"], "*_identified.json"))):
        with open(f) as fh:
            result = json.load(fh)
        for b in result.get("birds", []):
            top = b.get("top3") or [{"label": b.get("label"), "p": b.get("confidence")}]
            rows.append({"photo": result["photo"], "bird_id": b["id"], "predicted": top[0]["label"],
                         "identified": b["identified"], "confidence": round(float(b["confidence"]), 4),
                         "detector_score": b.get("detector_score"),
                         "box": " ".join(str(v) for v in b.get("box", [])),
                         "top3": "; ".join(f"{t['label']} {t['p']:.2f}" for t in top),
                         "result_file": os.path.basename(f)})
    df = pd.DataFrame(rows, columns=COLUMNS)
    log.info("%d birds in %d field photos", len(df), df["photo"].nunique())
    return FIELD_PREDICTIONS.validate(df) if len(df) else df


def _key(df):
    return df["photo"].astype(str) + "#" + df["bird_id"].astype(str)


def build_review_queue(predictions, paths, r):
    """Low-confidence and unidentified birds, lowest confidence first. Rows a reviewer already
    filled in (correct_label) are carried over, so re-running never loses review work."""
    queue = review_queue(predictions, r["min_confidence"]).astype(str)
    path = paths["review_queue"]
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "correct_label" in old:
            done = old[old["correct_label"].str.strip() != ""]
            queue = pd.concat([done, queue[~_key(queue).isin(set(_key(done)))]], ignore_index=True)
    log.info("Review queue: %d birds (%d already reviewed)", len(queue),
             int((queue["correct_label"].fillna("").astype(str).str.strip() != "").sum()) if len(queue) else 0)
    return queue.reindex(columns=COLUMNS + REVIEW_COLUMNS).fillna("")


def monitoring_report(predictions, queue, image_index, version, r, m):
    version = as_version(version)
    photos = [p for p in predictions["photo"].unique() if os.path.exists(p)]
    field = index_images([(p, "") for p in photos]) if photos else pd.DataFrame(columns=r["drift_features"])
    train = image_index[image_index["readable"].astype(bool)]
    if len(field):
        field = field[field["readable"].astype(bool)]
    drift = drift_report(train, field, r["drift_features"], r["psi_warn"], r["psi_alert"], r.get("min_photos", 30))

    tests = os.path.join("data", "pipeline", version, "test_predictions.csv")
    conf = {"field_mean": float(predictions["confidence"].mean()) if len(predictions) else None}
    if os.path.exists(tests) and len(predictions):
        test_conf = pd.read_csv(tests)["confidence"]
        conf.update(test_mean=float(test_conf.mean()), psi=psi(test_conf, predictions["confidence"]))
        conf["level"] = ("alert" if conf["psi"] > r["psi_alert"] else "warn" if conf["psi"] > r["psi_warn"]
                         else "stable")

    filled = queue["correct_label"].fillna("").astype(str).str.strip() if len(queue) else None
    reviewed = queue[filled != ""] if len(queue) else queue
    confirmed = int((reviewed["correct_label"] == reviewed["predicted"]).sum()) if len(reviewed) else 0
    report = {
        "version": version,
        "field_photos": int(predictions["photo"].nunique()), "birds": int(len(predictions)),
        "unidentified_share": float((~predictions["identified"].astype(bool)).mean()) if len(predictions) else None,
        "model_confidence": conf,
        "review": {"queued": int(len(queue)), "reviewed": int(len(reviewed)), "confirmed": confirmed,
                   "corrected": int(len(reviewed)) - confirmed,
                   "accuracy_on_reviewed": confirmed / len(reviewed) if len(reviewed) else None,
                   "note": "the queue holds the least certain birds, so this accuracy is a lower bound"},
        "photo_drift": drift,
    }
    log.info("Field photos %d, birds %d, photo drift %s, confidence drift %s", report["field_photos"],
             report["birds"], drift["overall"], conf.get("level", "n/a"))
    if m.get("enabled") and len(predictions):
        _log_monitoring(report, m)
    return report


def _log_monitoring(report, m):
    import mlflow

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment"])
    with mlflow.start_run(run_name=f"monitor {report['version']}"):
        mlflow.set_tags({"birds_version": report["version"], "stage": "monitoring"})
        metrics = {"field_photos": report["field_photos"], "field_birds": report["birds"],
                   "unidentified_share": report["unidentified_share"],
                   "field_mean_confidence": report["model_confidence"]["field_mean"],
                   "confidence_psi": report["model_confidence"].get("psi"),
                   "reviewed_birds": report["review"]["reviewed"],
                   "accuracy_on_reviewed": report["review"]["accuracy_on_reviewed"]}
        metrics.update({f"psi_{f['feature']}": f["psi"] for f in report["photo_drift"]["features"]})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.log_dict(report, "monitoring_report.json")
