"""Monitoring after deployment: data drift and a human review queue.

``drift_report`` compares simple image statistics of new photos with the training photos
using the population stability index (PSI). ``review_queue`` lists predictions a person
should check, lowest confidence first; the corrected labels go back into training.
"""

import numpy as np
import pandas as pd


def psi(reference, current, bins=10):
    """Population stability index between two samples of one numeric feature.
    Rule of thumb: < 0.1 stable, 0.1-0.25 some drift, > 0.25 strong drift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # reference (almost) constant: use even bins over both samples instead
        lo, hi = min(reference.min(), current.min()), max(reference.max(), current.max())
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference, current, features, warn=0.1, alert=0.25, min_current=30):
    rows = []
    for f in features:
        value = psi(reference[f], current[f])
        level = "n/a" if value is None else "alert" if value > alert else "warn" if value > warn else "stable"
        rows.append({"feature": f, "psi": value, "level": level,
                     "reference_mean": float(np.nanmean(reference[f])) if len(reference) else None,
                     "current_mean": float(np.nanmean(current[f])) if len(current) else None})
    levels = [r["level"] for r in rows]
    overall = "alert" if "alert" in levels else "warn" if "warn" in levels else "stable"
    note = ""
    if len(current) < min_current:
        note = f"only {len(current)} new images; PSI is unreliable below {min_current}, so treat levels as indicative"
    return {"overall": overall, "reference_images": int(len(reference)), "current_images": int(len(current)),
            "thresholds": {"warn": warn, "alert": alert}, "note": note, "features": rows}


def review_queue(predictions, min_confidence=0.5, include_unidentified=True):
    """predictions: one row for each prediction with at least 'confidence'.
    Returns the rows to check, lowest confidence first, with empty columns for the reviewer."""
    mask = predictions["confidence"] < min_confidence
    if include_unidentified and "identified" in predictions:
        mask |= ~predictions["identified"].astype(bool)
    queue = predictions[mask].sort_values("confidence").copy()
    queue["correct_label"] = ""
    queue["reviewer"] = ""
    queue["reviewed_on"] = ""
    return queue.reset_index(drop=True)
