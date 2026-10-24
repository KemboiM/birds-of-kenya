"""Data-validation checks for an image dataset with one folder for each label.

Each check returns a ``CheckResult``: its name, whether it passed, how many problems it
found and a table of the problem rows. ``summarize`` turns the results into a short report
and says whether any blocking check failed.

Nothing here is specific to birds. The checks work for any image classification dataset laid
out as ``<image_dir>/<label>/<file>``, with optional Pascal VOC box files and a metadata
table that records the license of each image.
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ISSUE_COLUMNS = ["check", "file", "label", "detail"]


@dataclass
class CheckResult:
    name: str
    passed: bool
    issues: int
    note: str = ""
    skipped: bool = False
    rows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ISSUE_COLUMNS))


def result(name, rows, note="", skipped=False):
    df = pd.DataFrame(rows, columns=["file", "label", "detail"])
    df.insert(0, "check", name)
    return CheckResult(name, not rows, len(rows), note=note, skipped=skipped, rows=df)


# ---------------------------------------------------------------- image index

def describe_image(path, label="", root=None):
    """One row for each image: content hash, readability, size and light statistics."""
    row = {"file": os.path.relpath(path, root) if root else path, "label": label,
           "bytes": os.path.getsize(path)}
    with open(path, "rb") as fh:
        row["sha1"] = hashlib.sha1(fh.read()).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            row["width"], row["height"] = im.size        # stored size, as the training code reads it
            row["exif_orientation"] = int(im.getexif().get(0x0112, 1) or 1)
            small = im.convert("RGB")
            small.thumbnail((128, 128))
            stat = ImageStat.Stat(small.convert("L"))
            row["brightness"], row["contrast"] = stat.mean[0], stat.stddev[0]
        row["aspect_ratio"] = row["width"] / row["height"]
        row["readable"], row["error"] = True, ""
    except Exception as exc:  # unreadable or truncated file
        row.update(readable=False, width=np.nan, height=np.nan, exif_orientation=1, brightness=np.nan,
                   contrast=np.nan, aspect_ratio=np.nan, error=str(exc)[:200])
    return row


def list_labelled_images(image_dir, max_per_label=None):
    """[(path, label)] for ``image_dir/<label>/<file>``."""
    items = []
    for label in sorted(os.listdir(image_dir)):
        folder = os.path.join(image_dir, label)
        if label.startswith(".") or not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
        items += [(os.path.join(folder, f), label) for f in (files[:max_per_label] if max_per_label else files)]
    return items


def index_images(items, root=None, workers=8):
    """Index [(path, label)] in parallel; returns one row for each image."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda it: describe_image(it[0], it[1], root), items))
    return pd.DataFrame(rows)


def _readable(index):
    return index[index["readable"].astype(bool)]


# ---------------------------------------------------------------- checks

def check_readable(index):
    bad = index[~index["readable"].astype(bool)]
    return result("unreadable_images", [(r.file, r.label, r.error) for r in bad.itertuples()])


def check_min_size(index, min_width, min_height):
    ok = _readable(index)
    small = ok[(ok["width"] < min_width) | (ok["height"] < min_height)]
    return result("too_small_images", [(r.file, r.label, f"{int(r.width)}x{int(r.height)} px")
                                       for r in small.itertuples()],
                  note=f"smaller than {min_width}x{min_height} px")


def check_duplicates(index):
    """Identical files. Same label: wasted, and a leak if they land in different splits
    (see check_splits). Different labels: one photo carries two labels, a labelling error."""
    same, cross = [], []
    for _, group in index.groupby("sha1"):
        if len(group) < 2:
            continue
        labels = sorted(set(group["label"]))
        target = cross if len(labels) > 1 else same
        for r in group.itertuples():
            target.append((r.file, r.label, f"same content as {len(group) - 1} other file(s); "
                                             f"labels {', '.join(labels)}"))
    return [result("duplicate_images", same), result("conflicting_labels", cross)]


def check_label_counts(index, min_per_label):
    counts = _readable(index).groupby("label").size()
    low = counts[counts < min_per_label]
    return result("labels_with_too_few_images",
                  [("", label, f"{n} usable images (need {min_per_label})") for label, n in low.items()],
                  note="these labels are left out of training")


def check_licenses(index, metadata, allowed, normalize=None, file_col="file", license_col="license"):
    """Every image must be listed in the metadata table with an allowed license."""
    if metadata is None or len(metadata) == 0:
        return result("licenses", [("", "", "metadata table missing or empty")])
    codes = metadata[license_col].map(normalize) if normalize else metadata[license_col]
    known = dict(zip(metadata[file_col].astype(str).map(os.path.normpath), codes))
    rows = []
    for r in index.itertuples():
        code = known.get(os.path.normpath(r.file))
        if code is None:
            rows.append((r.file, r.label, "not in metadata (no license record)"))
        elif code not in allowed:
            rows.append((r.file, r.label, f"license {code} not allowed"))
    return result("licenses", rows, note="allowed: " + ", ".join(allowed))


def read_voc(xml_path):
    """(width, height, [(name, xmin, ymin, xmax, ymax), ...]) from a Pascal VOC file."""
    root = ET.parse(xml_path).getroot()
    w, h = int(root.find("size/width").text), int(root.find("size/height").text)
    objects = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        objects.append((obj.find("name").text,
                        *(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return w, h, objects


def check_annotations(index, annotation_dir, tolerance_px=2):
    """Box files: the label must match the folder, boxes must lie inside the photo and have an
    area, and the size recorded in the file must match the photo (boxes are scaled by it).
    Photos without a box file are counted, not failed."""
    rows, with_boxes, missing = [], 0, 0
    for r in _readable(index).itertuples():
        stem = os.path.splitext(os.path.basename(r.file))[0]
        xml_path = os.path.join(annotation_dir, r.label, stem + ".xml")
        if not os.path.exists(xml_path):
            missing += 1
            continue
        with_boxes += 1
        try:
            w, h, objects = read_voc(xml_path)
        except Exception as exc:
            rows.append((r.file, r.label, f"box file unreadable: {exc}"))
            continue
        iw, ih = int(r.width), int(r.height)
        orientation = int(getattr(r, "exif_orientation", 1) or 1)
        if orientation != 1:
            # the box was drawn on the photo turned upright, but training reads the stored pixels
            rows.append((r.file, r.label, f"photo has EXIF orientation {orientation}: boxes drawn on the turned "
                                          f"photo do not line up with the stored pixels training reads"))
        elif (w, h) != (iw, ih):
            rows.append((r.file, r.label, f"box file says {w}x{h}, photo is {iw}x{ih}"))
        t = tolerance_px
        for name, x1, y1, x2, y2 in objects:
            if name != r.label:
                rows.append((r.file, r.label, f"box labelled {name}"))
            if not (-t <= x1 < x2 <= w + t and -t <= y1 < y2 <= h + t):
                rows.append((r.file, r.label, f"box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) outside the photo or empty"))
    return result("annotations", rows, note=f"{with_boxes} photos with box files, {missing} without "
                                            f"(used whole)")


def check_splits(index, splits, require_all_labels=True):
    """splits: {"train": [(file, label), ...], "val": [...], "test": [...]}.

    Fails if a photo sits in two splits, if the same content appears in train and test under
    different file names, if a listed photo is not in the dataset, or if a split lacks a label."""
    if not splits:
        return result("split_integrity", [], note="no split lists for this version yet", skipped=True)
    rows = []
    by_file = index.drop_duplicates("file").set_index(index.drop_duplicates("file")["file"].map(os.path.normpath))
    split_of, hashes, labels_by_split = {}, {}, {}
    for split, items in splits.items():
        labels_by_split[split] = set()
        for f, label in dict.fromkeys(items):           # crop splits list a photo once for each box
            f = os.path.normpath(f)
            labels_by_split[split].add(label)
            if f in split_of and split_of[f] != split:
                rows.append((f, label, f"in both {split_of[f]} and {split}"))
            split_of[f] = split
            if f not in by_file.index:
                rows.append((f, label, f"listed in {split} but not in the dataset"))
                continue
            hashes.setdefault(by_file.at[f, "sha1"], set()).add(split)
    leaked = {h for h, s in hashes.items() if "train" in s and "test" in s}
    for f, split in split_of.items():
        if split == "test" and f in by_file.index and by_file.at[f, "sha1"] in leaked:
            rows.append((f, by_file.at[f, "label"], "identical photo also in train"))
    if require_all_labels:
        all_labels = set().union(*labels_by_split.values())
        for split, labels in labels_by_split.items():
            rows += [("", label, f"label missing from {split}") for label in sorted(all_labels - labels)]
    n = {s: len(dict.fromkeys(v)) for s, v in splits.items()}
    return result("split_integrity", rows, note=", ".join(f"{s} {k} photos" for s, k in n.items()))


# ---------------------------------------------------------------- summary

def summarize(results, blocking):
    """Short report plus one table of every problem row. A failed check in ``blocking``
    fails the whole run; other failed checks are warnings."""
    checks = [{"check": r.name, "passed": bool(r.passed), "issues": int(r.issues),
               "blocking": r.name in blocking, "skipped": bool(r.skipped), "note": r.note} for r in results]
    failed = [c["check"] for c in checks if not c["passed"] and c["blocking"]]
    warnings = [c["check"] for c in checks if not c["passed"] and not c["blocking"]]
    summary = {"passed": not failed, "blocking_failures": failed, "warnings": warnings, "checks": checks}
    frames = [r.rows for r in results if len(r.rows)]
    issues = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ISSUE_COLUMNS)
    return summary, issues
