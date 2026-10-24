"""Promotion: the evaluated version replaces the one in production only when the gate passes.
Production is recorded twice: the MLflow alias "production" and reports/production.json."""

import logging
import os
from datetime import date

import pandas as pd

from ...project import as_version, read_json, train_counts
from ...reliability.evaluation import report_from_saved
from ...reliability.gate import promotion_decision

log = logging.getLogger(__name__)
PRODUCTION_FILE = os.path.join("reports", "production.json")


def _mlflow_production(m):
    if not m.get("enabled"):
        return None
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(m["tracking_uri"])
    try:
        mv = MlflowClient().get_model_version_by_alias(m["registered_model"], "production")
    except MlflowException:
        return None
    return mv.tags.get("birds_version"), int(mv.version)


def resolve_baseline(p, m):
    """(version, where it came from) of the version the candidate must beat."""
    if str(p["baseline"]) != "production":
        return as_version(p["baseline"]), "set in parameters.yml"
    record = read_json(PRODUCTION_FILE, {})   # committed with the code, so it is the record of what is in use
    if record.get("version"):
        return record["version"], PRODUCTION_FILE
    found = _mlflow_production(m)
    if found and found[0]:
        return found[0], f"MLflow alias production (registered version {found[1]})"
    return as_version(p["fallback_baseline"]), "fallback_baseline: nothing is in production yet"


def load_report(version, paths, rare_max_train):
    """Evaluation report of an earlier version: the Kedro report if there is one, otherwise
    built from the files main.py wrote to reports/<version>/."""
    folder = os.path.join("reports", version)
    report = read_json(os.path.join(folder, "evaluation_report.json"))
    if report and os.path.exists(os.path.join(folder, "evaluation_per_species.csv")):
        return report, pd.read_csv(os.path.join(folder, "evaluation_per_species.csv"))
    metrics = read_json(os.path.join(folder, "metrics.json"))
    if metrics is None or not os.path.exists(os.path.join(folder, "per_species_2x2.csv")):
        raise FileNotFoundError(f"No evaluation report for baseline version {version} in {folder}/")
    try:
        counts = train_counts(paths, version)
    except FileNotFoundError:
        counts = None
    return report_from_saved(metrics, pd.read_csv(os.path.join(folder, "per_species_2x2.csv")), counts, rare_max_train)


def decide(report, per_label, data_checks, version, paths, p, e, m):
    version = as_version(version)
    baseline_version, baseline_source = resolve_baseline(p, m)
    if baseline_version == version:
        baseline, baseline_per_label = None, None
        note = f"version {version} is already the baseline ({baseline_source})"
    else:
        baseline, baseline_per_label = load_report(baseline_version, paths, e["rare_max_train"])
        note = f"compared with version {baseline_version} ({baseline_source})"
    rules = {k: v for k, v in p.items() if k not in ("baseline", "fallback_baseline")}
    decision = promotion_decision(report, per_label, baseline, baseline_per_label, data_checks, **rules)
    decision = {"candidate": version, "promote": decision["promote"],
                "baseline": {"version": baseline_version, "source": baseline_source},
                "decided_on": date.today().isoformat(), "reasons": [note] + decision["reasons"],
                "rules": rules, "checks": decision["checks"]}
    log.info("Promote %s: %s", version, "YES" if decision["promote"] else "NO")
    for r in decision["reasons"]:
        log.info("  %s", r)
    return decision


def apply_decision(decision, mlflow_run, m):
    """Move the production alias when the gate passed; otherwise keep the record as it is."""
    current = read_json(PRODUCTION_FILE, {}) or {"version": None}
    version = decision["candidate"]
    if not decision["promote"]:
        log.info("Production stays on %s", current.get("version"))
        return current
    if current.get("version") == version:
        return current
    record = {"version": version, "promoted_on": decision["decided_on"],
              "replaces": current.get("version"), "decision": f"reports/{version}/promotion_decision.json"}
    mv = (mlflow_run or {}).get("model_version")
    if m.get("enabled") and mv:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(m["tracking_uri"])
        MlflowClient().set_registered_model_alias(m["registered_model"], "production", str(mv))
        record.update(mlflow_model=m["registered_model"], mlflow_model_version=mv)
        log.info("MLflow alias production -> %s version %s", m["registered_model"], mv)
    elif m.get("enabled"):
        _clear_mlflow_alias(m, version)
    log.info("Production is now version %s", version)
    return record


def _clear_mlflow_alias(m, version):
    """The promoted version has no registered model (its model file is not on this computer), so an
    older model must not keep the production alias."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(m["tracking_uri"])
    try:
        MlflowClient().delete_registered_model_alias(m["registered_model"], "production")
        log.warning("Version %s has no registered model, so the MLflow production alias was removed; "
                    "reports/production.json records the change", version)
    except MlflowException:
        pass
