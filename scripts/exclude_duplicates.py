#!/usr/bin/env python3
"""Add duplicate photos to data/annotations/exclude.csv so training leaves them out.

GBIF sometimes attaches one photo to observations of several species, for example a mixed
flock photographed once and recorded once for each species. The same pixels then carry two
labels, so every copy is left out. A photo saved twice under one species keeps its first copy;
the others are left out so that copies cannot end up in both the training and the test photos.

Rows already in exclude.csv are kept, and running the script again adds nothing new.

    python3 scripts/exclude_duplicates.py
    python3 scripts/exclude_duplicates.py --index data/pipeline/2.1/image_index.csv   # reuse the hashes
                                                                                      # from the data checks
"""

import argparse
import csv
import hashlib
import os
from collections import defaultdict

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def photo_hashes(data_root):
    """{content hash: [(path relative to data_root, species), ...]}"""
    image_root = os.path.join(data_root, "images", "sorted_data")
    groups = defaultdict(list)
    for species in sorted(os.listdir(image_root)):
        folder = os.path.join(image_root, species)
        if species.startswith(".") or not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(EXTENSIONS):
                path = os.path.join(folder, name)
                with open(path, "rb") as f:
                    digest = hashlib.sha1(f.read()).hexdigest()
                groups[digest].append((os.path.relpath(path, data_root), species))
    return groups


def hashes_from_index(path):
    """Same groups, read from the image index the Kedro data checks write (file, label, sha1)."""
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sha1"):
                groups[row["sha1"]].append((row["file"], row["label"]))
    return groups


def main():
    p = argparse.ArgumentParser(description="List duplicate photos in exclude.csv.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default=os.path.join("data", "annotations", "exclude.csv"))
    p.add_argument("--index", help="image_index.csv from the data checks, instead of reading every photo")
    args = p.parse_args()

    existing = []
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            existing = list(csv.DictReader(f))
    listed = {os.path.normpath(r["image"]) for r in existing if r.get("image")}

    new, mixed, repeats = [], 0, 0
    groups = hashes_from_index(args.index) if args.index else photo_hashes(args.data_root)
    for files in groups.values():
        files = sorted(files)
        if len(files) < 2:
            continue
        species = sorted({s for _, s in files})
        if len(species) > 1:
            drop, reason = files, "same photo filed under " + ", ".join(species)
            mixed += 1
        else:
            drop, reason = files[1:], f"duplicate of {files[0][0]}"
            repeats += 1
        for rel, _ in drop:
            if os.path.normpath(rel) not in listed:
                new.append({"image": rel, "reason": reason})
                listed.add(os.path.normpath(rel))

    extra = [k for k in (existing[0] if existing else {}) if k not in ("image", "reason")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "reason", *extra], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + new)
    print(f"{mixed} photos filed under more than one species, {repeats} photos saved twice under one species")
    print(f"{len(new)} photos added to {args.out} ({len(existing) + len(new)} listed in all)")


if __name__ == "__main__":
    main()
