"""Where this repository keeps things, and small readers for its own file formats."""

import csv
import json
import os

PROJECT = "birds"  # file-name prefix used by main.py and identify.py


def model_path(paths, version):
    return os.path.join(paths["checkpoints"], version, f"{PROJECT}_classify_{version}.keras")


def label_list_path(paths, version):
    return os.path.join(paths["labels"], f"{PROJECT}_label_list_{version}.txt")


def read_label_list(paths, version):
    with open(label_list_path(paths, version)) as f:
        return [line.strip() for line in f if line.strip()]


def read_split(paths, version, split):
    """[(file relative to data_root, label, box or None)] from splits/<split>_birds_<version>.txt.
    box = [ymin, xmin, ymax, xmax] as fractions of the photo (crop-trained versions)."""
    path = os.path.join(paths["splits"], f"{split}_{PROJECT}_{version}.txt")
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            box = [float(v) for v in parts[2].split()] if len(parts) > 2 and parts[2].strip() else None
            rows.append((parts[0], parts[1], box))
    return rows


def read_splits(paths, version):
    """{"train": [(file, label)], "val": [...], "test": [...]}, or {} if this version has no split lists."""
    out = {}
    for split in ("train", "val", "test"):
        try:
            out[split] = [(f, label) for f, label, _ in read_split(paths, version, split)]
        except FileNotFoundError:
            return {}
    return out


def train_counts(paths, version):
    """Training photos for each label (a photo listed once for each box is counted once)."""
    counts, seen = {}, set()
    for f, label, _ in read_split(paths, version, "train"):
        if f not in seen:
            seen.add(f)
            counts[label] = counts.get(label, 0) + 1
    return counts


def read_exclude(paths):
    """Photos left out of training (data/annotations/exclude.csv, column "image")."""
    if not paths.get("exclude") or not os.path.exists(paths["exclude"]):
        return set()
    with open(paths["exclude"], newline="") as f:
        return {os.path.normpath(row["image"]) for row in csv.DictReader(f) if row.get("image")}


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def as_version(value):
    """Versions are names, not numbers: keep "1.10" as written. A bare float from the command
    line (version=1.2) is turned back into text."""
    return value if isinstance(value, str) else repr(value)
