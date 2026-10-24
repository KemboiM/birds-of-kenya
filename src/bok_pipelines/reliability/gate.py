"""Promotion gate: a new model replaces the one in use only when it is better.

Compares a candidate evaluation report with a baseline report (normally the model in use) and returns
a decision with every reason written out, so the decision can be read and audited later.
"""


def _check(checks, reasons, name, passed, text, **values):
    checks[name] = {**values, "passed": bool(passed)}
    reasons.append(("PASS " if passed else "FAIL ") + text)


def promotion_decision(candidate, candidate_per_label, baseline=None, baseline_per_label=None,
                       data_checks=None, require_data_checks=True, metric="accuracy", min_gain=0.0,
                       max_macro_f1_drop=0.0, max_rare_accuracy_drop=0.0, max_label_recall_drop=0.20,
                       min_test_images=15, max_regressed_labels=0, **_):
    """candidate / baseline: summary dicts with at least `metric` and f1_macro
    (accuracy_rare_labels is compared when both have it).
    *_per_label: DataFrames with label, test_images and recall."""
    reasons, checks = [], {}

    if data_checks is None:
        _check(checks, reasons, "data_checks", not require_data_checks,
               "data checks were not run for this version", required=require_data_checks)
    elif data_checks.get("subset"):
        _check(checks, reasons, "data_checks", False, "data checks ran on a trial subset, not on every image",
               subset=data_checks["subset"])
    else:
        failed = data_checks.get("blocking_failures", [])
        _check(checks, reasons, "data_checks", data_checks.get("passed", False),
               "data checks passed" if not failed else "data checks failed: " + ", ".join(failed),
               blocking_failures=failed)

    if baseline is None:
        reasons.append("no other version to compare with, so only the data checks decide")
        return {"promote": all(c["passed"] for c in checks.values()), "reasons": reasons, "checks": checks}

    gain = candidate[metric] - baseline[metric]
    _check(checks, reasons, "metric_gain", gain >= min_gain,
           f"{metric} {candidate[metric]:.4f} vs {baseline[metric]:.4f} for the baseline "
           f"(change {gain:+.4f}, required at least {min_gain:+.4f})",
           metric=metric, candidate=candidate[metric], baseline=baseline[metric], gain=gain, required=min_gain)

    drop = baseline["f1_macro"] - candidate["f1_macro"]
    _check(checks, reasons, "macro_f1", drop <= max_macro_f1_drop,
           f"macro F1 {candidate['f1_macro']:.4f} vs {baseline['f1_macro']:.4f} "
           f"(fall {drop:+.4f}, allowed {max_macro_f1_drop:+.4f})",
           candidate=candidate["f1_macro"], baseline=baseline["f1_macro"], drop=drop, allowed=max_macro_f1_drop)

    c_rare, b_rare = candidate.get("accuracy_rare_labels"), baseline.get("accuracy_rare_labels")
    if c_rare is not None and b_rare is not None:
        drop = b_rare - c_rare
        _check(checks, reasons, "rare_labels", drop <= max_rare_accuracy_drop,
               f"accuracy on rare labels ({candidate.get('rare_label_rule', 'rare')}) {c_rare:.4f} vs {b_rare:.4f} "
               f"(fall {drop:+.4f}, allowed {max_rare_accuracy_drop:+.4f})",
               candidate=c_rare, baseline=b_rare, drop=drop, allowed=max_rare_accuracy_drop)

    if candidate_per_label is not None and baseline_per_label is not None:
        base = baseline_per_label.drop_duplicates("label").set_index("label")["recall"]
        regressed = []
        for r in candidate_per_label.itertuples():
            if r.label in base.index and r.test_images >= min_test_images:
                fall = float(base[r.label]) - float(r.recall)
                if fall > max_label_recall_drop + 1e-9:
                    regressed.append({"label": r.label, "test_images": int(r.test_images),
                                      "baseline_recall": round(float(base[r.label]), 4),
                                      "candidate_recall": round(float(r.recall), 4), "fall": round(fall, 4)})
        regressed.sort(key=lambda x: -x["fall"])
        _check(checks, reasons, "label_regressions", len(regressed) <= max_regressed_labels,
               f"{len(regressed)} label(s) lost more than {max_label_recall_drop:.0%} recall "
               f"(labels with {min_test_images}+ test images; allowed {max_regressed_labels})",
               labels=regressed, allowed=max_regressed_labels)

    return {"promote": all(c["passed"] for c in checks.values()), "reasons": reasons, "checks": checks}
