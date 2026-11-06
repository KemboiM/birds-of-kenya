#!/usr/bin/env python3
"""
List the species_info/ files to check against a field guide, most important first.

    python3 scripts/species_review_list.py --version 2.1

Writes species_info/review_priority.csv. Order: files not yet verified, then by
how often the species is recorded in Kenya on GBIF (the birds people photograph
most come first). The "problems" column says what to look at; distance estimates
depend on the adult size, so size problems matter most.
"""

import argparse
import csv
import json
import os


def size_problems(size):
    adult = (size or {}).get("adult") or {}
    length = adult.get("length_cm")
    if not length:
        return ["no adult length (no distance estimate)"]
    lo, hi = min(length), max(length)
    out = []
    if lo < 7 or hi > 300:
        out.append(f"adult length {lo:g}-{hi:g} cm looks wrong")
    elif hi / max(lo, 1e-6) > 1.6:
        out.append(f"adult length range {lo:g}-{hi:g} cm too wide")
    if not adult.get("wingspan_cm"):
        out.append("no wingspan")
    return out


def main():
    p = argparse.ArgumentParser(description="Make the species_info review list.")
    p.add_argument("--version", default="2.1")
    p.add_argument("--info", default="species_info")
    args = p.parse_args()

    labels_path = os.path.join("model", "configs_and_labels", f"birds_label_list_{args.version}.txt")
    with open(labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]
    f1 = {}
    pss = os.path.join("data", "validation_data", args.version, f"{args.version}_results", "per_species_2x2.csv")
    if os.path.exists(pss):
        with open(pss, newline="") as f:
            f1 = {r["species"]: r for r in csv.DictReader(f)}

    rows = []
    for label in labels:
        path = os.path.join(args.info, f"{label}.json")
        if not os.path.exists(path):
            rows.append({"species": label, "common_name": "", "verified_by": "", "gbif_records_in_kenya": 0,
                         "test_photos": f1.get(label, {}).get("test_images", ""),
                         "test_f1": f1.get(label, {}).get("f1", ""),
                         "problems": "no species file yet: run scripts/build_species_info.py"})
            continue
        with open(path) as f:
            e = json.load(f)
        problems = []
        if not e.get("common_name"):
            problems.append("no common name")
        problems += size_problems(e.get("size"))
        if not e.get("summary"):
            problems.append("no description")
        la = e.get("look_alikes") or []
        if la and not all(x.get("difference") for x in la):
            problems.append(f"{sum(1 for x in la if not x.get('difference'))} look-alike(s) without the difference written")
        rows.append({
            "species": label,
            "common_name": e.get("common_name") or "",
            "verified_by": (e.get("review") or {}).get("verified_by") or "",
            "gbif_records_in_kenya": (e.get("where_found") or {}).get("gbif_records_in_kenya") or 0,
            "test_photos": f1.get(label, {}).get("test_images", ""),
            "test_f1": f1.get(label, {}).get("f1", ""),
            "problems": "; ".join(problems),
        })

    rows.sort(key=lambda r: (bool(r["verified_by"]), -int(r["gbif_records_in_kenya"] or 0)))
    for i, r in enumerate(rows, 1):
        r["priority"] = i
    out = os.path.join(args.info, "review_priority.csv")
    fields = ["priority", "species", "common_name", "verified_by", "gbif_records_in_kenya",
              "test_photos", "test_f1", "problems"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    todo = [r for r in rows if not r["verified_by"]]
    print(f"{out}: {len(rows)} species, {len(todo)} not verified, "
          f"{sum(1 for r in rows if 'no species file' in r['problems'])} without a file, "
          f"{sum(1 for r in rows if 'length' in r['problems'])} with a size problem")


if __name__ == "__main__":
    main()
