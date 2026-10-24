"""Data checks before training: is every photo readable, licensed, correctly boxed and
in exactly one split?"""

import logging
import os

import pandas as pd

from ...project import as_version, read_exclude, read_splits
from ...reliability import data_checks as dc
from ...reliability.schemas import METADATA, schema_issues

log = logging.getLogger(__name__)


def build_image_index(paths, params):
    """Hash and measure every training photo (photos in exclude.csv are left out)."""
    items = dc.list_labelled_images(paths["images"], params.get("max_images_per_label"))
    exclude = read_exclude(paths)
    rel = lambda p: os.path.normpath(os.path.relpath(p, paths["data_root"]))
    kept = [it for it in items if rel(it[0]) not in exclude]
    log.info("Indexing %d photos in %s (%d excluded)", len(kept), paths["images"], len(items) - len(kept))
    return dc.index_images(kept, root=paths["data_root"], workers=params.get("workers", 8))


def _license_code():
    from scripts.download_gbif_images import PHOTO_LICENSES, photo_license_code
    codes = set().union(*PHOTO_LICENSES.values())
    return lambda text: text if text in codes else photo_license_code(text)


def run_checks(index, version, paths, data_params, training_params):
    version = as_version(version)
    results = []

    metadata = pd.read_csv(paths["metadata"], dtype=str) if os.path.exists(paths["metadata"]) else None
    if metadata is None:
        results.append(dc.result("metadata_table", [("", "", f"{paths['metadata']} not found")]))
    else:
        metadata, rows = schema_issues(METADATA, metadata, "metadata")
        results.append(dc.result("metadata_table", rows, note=f"{len(metadata)} rows"))

    results.append(dc.check_readable(index))
    results.append(dc.check_min_size(index, data_params["min_width"], data_params["min_height"]))
    results += dc.check_duplicates(index)
    results.append(dc.check_label_counts(index, data_params["min_images_per_label"]))
    results.append(dc.check_licenses(index, metadata, data_params["allowed_licenses"], _license_code()))
    results.append(dc.check_annotations(index, paths["annotations"], data_params.get("box_tolerance_px", 2)))

    split_version = data_params.get("splits_version") or (
        training_params["split_from"] if training_params.get("from_annotations") else version)
    exclude = read_exclude(paths)   # photos left out of training may still be listed in an older split
    splits = {s: [(f, label) for f, label in items if os.path.normpath(f) not in exclude]
              for s, items in read_splits(paths, as_version(split_version)).items()}
    subset = bool(data_params.get("max_images_per_label"))
    if subset:  # trial run on a subset: check only the photos indexed
        indexed = set(index["file"].map(os.path.normpath))
        splits = {s: [(f, label) for f, label in items if os.path.normpath(f) in indexed] for s, items in splits.items()}
    split_check = dc.check_splits(index, splits, require_all_labels=not subset)
    split_check.note = f"split of version {as_version(split_version)}: " + split_check.note
    results.append(split_check)

    summary, issues = dc.summarize(results, set(data_params["blocking"]))
    summary = {"version": version, "images_checked": int(len(index)), "excluded_photos": len(exclude),
               "labels": int(index["label"].nunique()) if len(index) else 0,
               "subset": data_params.get("max_images_per_label"), **summary}
    for c in summary["checks"]:
        state = "skipped" if c["skipped"] else "ok" if c["passed"] else "FAIL" if c["blocking"] else "warn"
        log.info("%-28s %-7s %6d issue(s)  %s", c["check"], state, c["issues"], c["note"])
    if not summary["passed"]:
        log.warning("Blocking data checks failed: %s. See data/pipeline/%s/data_check_issues.csv",
                    ", ".join(summary["blocking_failures"]), version)
    return summary, issues
