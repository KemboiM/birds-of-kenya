"""Standard evaluation report for a classifier, produced before the model is used.

From true labels and predicted probabilities it reports overall accuracy, top-k
accuracy, macro and weighted F1, calibration (expected calibration error), accuracy
at a confidence cut-off with the share of predictions it covers, results for rare and
common labels separately, and one row for each label.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def expected_calibration_error(confidence, correct, bins=10):
    """Average gap between confidence and accuracy, weighted by how many predictions fall in each bin."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def evaluation_report(y_true, probs, class_names, train_counts=None, rare_max_train=20,
                      confidence_cutoff=0.5, top_k=5, calibration_bins=10):
    """Returns (summary dict, DataFrame with one row for each label)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    labels = list(range(len(class_names)))
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    k = min(top_k, len(class_names))
    top = np.argsort(probs, axis=1)[:, -k:]

    summary = {
        "num_classes": len(class_names),
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"top{k}_accuracy": float(np.mean([t in row for t, row in zip(y_true, top)])),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "expected_calibration_error": expected_calibration_error(conf, correct, calibration_bins),
        f"accuracy_at_confidence_{confidence_cutoff}": float(correct[conf >= confidence_cutoff].mean())
        if (conf >= confidence_cutoff).any() else None,
        f"coverage_at_confidence_{confidence_cutoff}": float((conf >= confidence_cutoff).mean()),
    }

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_label = pd.DataFrame({"label": class_names, "test_images": support, "precision": prec,
                              "recall": rec, "f1": f1})
    if train_counts is not None:
        per_label["train_images"] = [int(train_counts.get(c, 0)) for c in class_names]
        rare = per_label["train_images"] <= rare_max_train
        index_rare = set(per_label.index[rare])
        is_rare = np.array([t in index_rare for t in y_true])
        summary["rare_label_rule"] = f"<= {rare_max_train} training images"
        summary["rare_labels"] = int(rare.sum())
        summary["accuracy_rare_labels"] = float(correct[is_rare].mean()) if is_rare.any() else None
        summary["accuracy_common_labels"] = float(correct[~is_rare].mean()) if (~is_rare).any() else None
    per_label = per_label.sort_values("f1").reset_index(drop=True)
    summary["worst_labels"] = per_label.head(10)[["label", "test_images", "recall", "f1"]].to_dict("records")
    return summary, per_label


def report_from_saved(metrics, per_label_csv, train_counts=None, rare_max_train=20):
    """Same report shape, built from evaluation files already written by training
    (metrics.json and a table with one row for each label: precision, recall, f1 and test-image counts)."""
    per_label = per_label_csv.rename(columns={"species": "label"}).copy()
    keep = [c for c in ("label", "test_images", "precision", "recall", "f1") if c in per_label]
    per_label = per_label[keep]
    summary = {k: v for k, v in metrics.items()}
    if train_counts is not None:
        per_label["train_images"] = per_label["label"].map(lambda c: int(train_counts.get(c, 0)))
        rare = per_label["train_images"] <= rare_max_train
        w = per_label["test_images"].astype(float)
        hits = per_label["recall"].astype(float) * w
        summary["rare_label_rule"] = f"<= {rare_max_train} training images"
        summary["rare_labels"] = int(rare.sum())
        summary["accuracy_rare_labels"] = float(hits[rare].sum() / w[rare].sum()) if w[rare].sum() else None
        summary["accuracy_common_labels"] = float(hits[~rare].sum() / w[~rare].sum()) if w[~rare].sum() else None
    per_label = per_label.sort_values("f1").reset_index(drop=True)
    summary["worst_labels"] = per_label.head(10)[["label", "test_images", "recall", "f1"]].to_dict("records")
    return summary, per_label
