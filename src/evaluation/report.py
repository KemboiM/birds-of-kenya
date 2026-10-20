"""End-of-training evaluation outputs, laid out like groundnut's validation_data.

Everything for one model version goes in one folder, e.g.
data/validation_data/1.0/1.0_results/:

    metrics.json                     overall accuracy, top-5, macro/weighted F1
    confusion_matrix.csv             N x N counts (rows = true, columns = predicted)
    confusion_matrix_normalized.csv  each row divided by its total (recall per cell)
    confusion_matrix.png             full matrix, or the most-confused species if N is large
    per_species_2x2.csv              TP / FN / FP / TN, precision, recall, F1 per species
    most_confused_pairs.csv          true -> predicted mix-ups, most frequent first
    misclassified.csv                every wrong test image with confidence and source link
    misclassified/<True>__predicted__<Pred>/<image>__conf0.62.jpg
    correct/<Species>/<image>__conf0.97.jpg      (only with copy_correct=True)
    training_curves.png              accuracy and loss per epoch
"""

import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def _load_source_links(data_root):
    """Map image path (relative to data root) -> GBIF occurrence link, if available."""
    links = {}
    path = os.path.join(data_root, "gbif_metadata.csv") if data_root else None
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                links[os.path.normpath(row["file"])] = row.get("occurrence_url", "")
    return links


def _write_matrix_csv(path, matrix, class_names, fmt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + class_names)
        for name, row in zip(class_names, matrix):
            w.writerow([name] + [fmt(v) for v in row])


def _plot_matrix(path, cm, class_names, max_classes):
    """Plot the row-normalized matrix; for large N keep the most-confused species."""
    n = len(class_names)
    idx = np.arange(n)
    title = f"Confusion matrix ({n} species)"
    if n > max_classes:
        off = cm.copy()
        np.fill_diagonal(off, 0)
        involvement = off.sum(axis=0) + off.sum(axis=1)
        idx = np.sort(np.argsort(involvement)[::-1][:max_classes])
        title = f"Confusion matrix: {max_classes} most-confused of {n} species"
    sub = cm[np.ix_(idx, idx)].astype(float)
    totals = cm[idx].sum(axis=1, keepdims=True).astype(float)
    norm = np.divide(sub, totals, out=np.zeros_like(sub), where=totals > 0)
    names = [class_names[i].replace("_", " ") for i in idx]

    size = max(6, 0.32 * len(idx) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(idx)), names, rotation=90, fontsize=7, style="italic")
    ax.set_yticks(range(len(idx)), names, fontsize=7, style="italic")
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title(title)
    if len(idx) <= 40:
        for r in range(len(idx)):
            for c in range(len(idx)):
                if sub[r, c]:
                    ax.text(c, r, int(sub[r, c]), ha="center", va="center", fontsize=6,
                            color="white" if norm[r, c] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of true species")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_history(path, history):
    h = history.history if hasattr(history, "history") else history
    if not h or "loss" not in h:
        return False
    epochs = range(1, len(h["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("accuracy", "train"), ("val_accuracy", "validation")):
        if key in h:
            a1.plot(epochs, h[key], label=label)
    for key, label in (("loss", "train"), ("val_loss", "validation")):
        if key in h:
            a2.plot(epochs, h[key], label=label)
    a1.set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    a2.set(title="Loss", xlabel="Epoch")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _copy(src, folder, confidence):
    os.makedirs(folder, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    dest = os.path.join(folder, f"{stem}__conf{confidence:.2f}{ext or '.jpg'}")
    shutil.copy2(src, dest)
    return dest


def write_evaluation(out_dir, y_true, probs, class_names, test_paths, history=None,
                     data_root=None, copy_correct=False, max_plot_classes=30):
    """Write all evaluation outputs to out_dir and return the overall metrics."""
    os.makedirs(out_dir, exist_ok=True)
    n = len(class_names)
    labels = list(range(n))
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    # ---- overall metrics
    k = min(5, n)
    top_k = np.argsort(probs, axis=1)[:, -k:]
    metrics = {
        "num_classes": n,
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"top{k}_accuracy": float(np.mean([t in row for t, row in zip(y_true, top_k)])),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "misclassified_images": int((y_true != y_pred).sum()),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- N x N matrices
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, totals, out=np.zeros(cm.shape), where=totals > 0)
    _write_matrix_csv(os.path.join(out_dir, "confusion_matrix.csv"), cm, class_names, int)
    _write_matrix_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"),
                      cm_norm, class_names, lambda v: f"{v:.3f}")
    _plot_matrix(os.path.join(out_dir, "confusion_matrix.png"), cm, class_names, max_plot_classes)

    # ---- per-species 2 x 2 (one species vs everything else)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    total = cm.sum()
    rows = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        rows.append([name, tp, fn, fp, tn, int(support[i]),
                     f"{prec[i]:.3f}", f"{rec[i]:.3f}", f"{f1[i]:.3f}"])
    rows.sort(key=lambda r: float(r[-1]))          # hardest species first
    with open(os.path.join(out_dir, "per_species_2x2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "TP", "FN", "FP", "TN", "test_images",
                    "precision", "recall", "f1"])
        w.writerows(rows)

    # ---- most-confused pairs
    pairs = [(class_names[t], class_names[p], int(cm[t, p]), cm_norm[t, p])
             for t in labels for p in labels if t != p and cm[t, p] > 0]
    pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
    with open(os.path.join(out_dir, "most_confused_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_species", "predicted_as", "times", "share_of_true_species"])
        w.writerows([(t, p, c, f"{s:.3f}") for t, p, c, s in pairs])

    # ---- image folders
    links = _load_source_links(data_root)
    with open(os.path.join(out_dir, "misclassified.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["copied_to", "original", "true_species", "predicted_as", "confidence",
                    "prob_of_true_species", "top3", "source"])
        for i, path in enumerate(test_paths):
            t, p = int(y_true[i]), int(y_pred[i])
            if t == p:
                if copy_correct:
                    _copy(path, os.path.join(out_dir, "correct", class_names[t]), conf[i])
                continue
            folder = os.path.join(out_dir, "misclassified",
                                  f"{class_names[t]}__predicted__{class_names[p]}")
            dest = _copy(path, folder, conf[i])
            top3 = "; ".join(f"{class_names[j]} {probs[i, j]:.2f}"
                             for j in np.argsort(probs[i])[::-1][:3])
            rel = os.path.normpath(os.path.relpath(path, data_root)) if data_root else path
            w.writerow([os.path.relpath(dest, out_dir), path, class_names[t], class_names[p],
                        f"{conf[i]:.3f}", f"{probs[i, t]:.3f}", top3, links.get(rel, "")])

    if history is not None:
        _plot_history(os.path.join(out_dir, "training_curves.png"), history)

    # ---- console summary
    print("\n=== Test Results ===")
    for key, value in metrics.items():
        print(f"{key:<22} {value:.4f}" if isinstance(value, float) else f"{key:<22} {value}")
    print("\nHardest species (lowest F1):")
    for r in rows[:5]:
        print(f"  {r[0]:<35} F1 {r[-1]}  (test images: {r[5]})")
    if pairs:
        print("\nMost-confused pairs (true -> predicted):")
        for t, p, c, s in pairs[:5]:
            print(f"  {t} -> {p}: {c} times ({s:.0%} of {t})")
    print(f"\nEvaluation outputs: {out_dir}")
    return metrics
