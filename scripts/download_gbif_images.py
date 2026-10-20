#!/usr/bin/env python3
"""
Download openly licensed photos of Kenyan birds from GBIF, one folder per species.

Source: GBIF occurrence API (country=KE, class Aves, field observations with photos).
Most photos come from iNaturalist research-grade observations.

Output (paths relative to --data-root, default "data"):
    images/sorted_data/<Scientific_name>/<Scientific_name>_1.jpg, _2.jpg, ...
    gbif_metadata.csv        one row per saved image: file, species, GBIF id,
                             license, photographer, date, coordinates, source link
    gbif_species_counts.csv  photo records available per species on GBIF

Re-running the same command resumes: photos already saved (matched by GBIF id
in gbif_metadata.csv) are skipped and numbering continues from the last file.

Only the Python standard library is needed (Python 3.8+).

Examples:
    # List every species and how many photos GBIF has; download nothing
    python3 scripts/download_gbif_images.py --dry-run

    # All Kenyan species with at least 20 photos, up to 200 each (the default)
    python3 scripts/download_gbif_images.py

    # Commercial-safe licenses only (CC0 and CC BY)
    python3 scripts/download_gbif_images.py --license commercial
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.gbif.org/v1"
AVES_TAXON_KEY = 212          # GBIF backbone key for class Aves
PAGE = 300                    # GBIF max page size for occurrence search
GBIF_OFFSET_CAP = 100_000     # GBIF search cannot page past this
USER_AGENT = "birds-of-kenya/1.0 (research; species identification tool)"

# GBIF filters by the *observation's* license; each photo can carry its own,
# so every photo is checked again below (photo_license_code).
LICENSES = {
    "commercial": ["CC0_1_0", "CC_BY_4_0"],                 # safe for any use
    "all": ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"],        # adds non-commercial
}
# Photo licenses accepted. No-derivatives (ND) and all-rights-reserved photos are
# never used, because training data is cropped, resized and augmented.
PHOTO_LICENSES = {
    "commercial": {"CC0", "CC-BY", "CC-BY-SA"},
    "all": {"CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA"},
}


def photo_license_code(text):
    """'http://creativecommons.org/licenses/by-nc/4.0/' -> 'CC-BY-NC'; None if not CC."""
    text = (text or "").lower()
    if "publicdomain/zero" in text or text.strip() in ("cc0_1_0", "cc0"):
        return "CC0"
    m = re.search(r"licenses/([a-z-]+)/", text) or re.match(r"cc_([a-z_]+?)_\d", text)
    return "CC-" + m.group(1).replace("_", "-").upper() if m else None


def is_photo(media):
    """Skip sound recordings and their spectrograms (e.g. xeno-canto), which GBIF lists as images."""
    url = media.get("identifier", "").lower()
    fmt = (media.get("format") or "").lower()
    if "xeno-canto" in url or "/sounds/" in url or "spectrogram" in url:
        return False
    return not fmt or fmt.startswith("image/jpeg") or fmt.startswith("image/jpg")

META_FIELDS = ["file", "species", "speciesKey", "gbifID", "url", "license",
               "rightsHolder", "eventDate", "lat", "lon", "stateProvince",
               "dataset", "occurrence_url"]


# ---------------------------------------------------------------- HTTP helpers

def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retrying in {wait}s ({e})", file=sys.stderr)
            time.sleep(wait)


def download(url, dest, retries=3):
    """Download one image to `dest`. Returns 'ok' or a reason it was skipped."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                ctype = r.headers.get("Content-Type", "")
                if "image" not in ctype:
                    return f"skipped (content-type {ctype or 'unknown'})"
                with open(dest, "wb") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
            return "ok"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                return f"failed ({e})"
            time.sleep(2 ** attempt)


def build_url(path, params):
    # list values become repeated keys, which GBIF treats as OR
    return f"{API}{path}?{urllib.parse.urlencode(params, doseq=True)}"


# ---------------------------------------------------------------- GBIF queries

def base_filters(args):
    return {
        "country": "KE",
        "taxonKey": AVES_TAXON_KEY,
        "mediaType": "StillImage",
        "basisOfRecord": "HUMAN_OBSERVATION",   # field photos, not museum skins
        "occurrenceStatus": "PRESENT",
        "license": LICENSES[args.license],
    }


def species_counts(args):
    """Occurrences-with-images per species, via a GBIF facet query."""
    params = dict(base_filters(args), limit=0, facet="speciesKey", facetLimit=5000)
    data = get_json(build_url("/occurrence/search", params))
    facets = data.get("facets") or []
    counts = facets[0]["counts"] if facets else []
    return [(int(c["name"]), int(c["count"])) for c in counts]


def species_name(key, cache={}):
    if key not in cache:
        sp = get_json(f"{API}/species/{key}")
        cache[key] = sp.get("canonicalName") or sp.get("scientificName") or str(key)
    return cache[key]


def safe_name(name):
    """'Haliaeetus vocifer' -> 'Haliaeetus_vocifer' (used for folders, labels, files)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")


def resize_url(url, size):
    """iNaturalist hosts several sizes; 'medium' (~500px) is plenty for 224px models."""
    if size == "original" or "inaturalist" not in url:
        return url
    return re.sub(r"/(original|large|medium|small|square)\.", f"/{size}.", url)


def image_records(species_key, args, skip_ids):
    """Yield one record per new image for a species, up to --max-per-species in total."""
    taken, offset = len(skip_ids), 0
    params = dict(base_filters(args), speciesKey=species_key, limit=PAGE)
    while taken < args.max_per_species and offset < GBIF_OFFSET_CAP:
        params["offset"] = offset
        data = get_json(build_url("/occurrence/search", params))
        for occ in data.get("results", []):
            allowed = PHOTO_LICENSES[args.license]
            stills = [m for m in occ.get("media", [])
                      if m.get("type") == "StillImage" and m.get("identifier") and is_photo(m)
                      and photo_license_code(m.get("license") or occ.get("license")) in allowed]
            # One photo per observation: extra photos of the same bird are
            # near-duplicates that would leak between train and test splits.
            for i, m in enumerate(stills if args.all_media else stills[:1]):
                gbif_id = str(occ.get("gbifID") or occ.get("key"))
                if i:
                    gbif_id = f"{gbif_id}-{i}"
                if gbif_id in skip_ids:
                    continue
                yield {
                    "gbifID": gbif_id,
                    "url": resize_url(m["identifier"], args.size),
                    "license": m.get("license") or occ.get("license", ""),
                    "rightsHolder": m.get("rightsHolder") or m.get("creator") or "",
                    "eventDate": occ.get("eventDate", ""),
                    "lat": occ.get("decimalLatitude", ""),
                    "lon": occ.get("decimalLongitude", ""),
                    "stateProvince": occ.get("stateProvince", ""),
                    "dataset": occ.get("datasetName", ""),
                    "occurrence_url": f"https://www.gbif.org/occurrence/{occ.get('key')}",
                }
                taken += 1
                if taken >= args.max_per_species:
                    return
        if data.get("endOfRecords", True):
            return
        offset += PAGE


def read_existing(meta_path):
    """Return {species_folder: (set of gbifIDs, highest file number)} from earlier runs."""
    done = {}
    if not os.path.exists(meta_path):
        return done
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            folder = os.path.basename(os.path.dirname(row["file"]))
            ids, top = done.get(folder, (set(), 0))
            ids.add(row["gbifID"])
            m = re.search(r"_(\d+)\.jpg$", row["file"])
            done[folder] = (ids, max(top, int(m.group(1)) if m else 0))
    return done


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data", help="project data folder (default: data)")
    p.add_argument("--top-species", type=int, default=0,
                   help="keep only the N species with the most photos (default 0 = all species)")
    p.add_argument("--min-per-species", type=int, default=20,
                   help="skip species with fewer photo records than this (default: 20)")
    p.add_argument("--max-per-species", type=int, default=200,
                   help="cap images per species to keep classes balanced (default: 200)")
    p.add_argument("--license", choices=LICENSES, default="all",
                   help="'commercial' = CC0, CC BY, CC BY-SA photos only; "
                        "'all' adds CC BY-NC and CC BY-NC-SA (default: all). ND photos are never used.")
    p.add_argument("--size", choices=["medium", "large", "original"], default="medium",
                   help="iNaturalist image size to fetch (default: medium, ~500px)")
    p.add_argument("--all-media", action="store_true",
                   help="take every photo per observation, not just the first")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (default: 8)")
    p.add_argument("--dry-run", action="store_true",
                   help="only list species and photo counts; download nothing")
    args = p.parse_args()

    image_root = os.path.join(args.data_root, "images", "sorted_data")
    meta_path = os.path.join(args.data_root, "gbif_metadata.csv")
    counts_path = os.path.join(args.data_root, "gbif_species_counts.csv")

    print("Querying GBIF for Kenyan bird species with photos...")
    counts = species_counts(args)
    counts = [(k, n) for k, n in counts if n >= args.min_per_species]
    counts.sort(key=lambda kn: kn[1], reverse=True)
    if args.top_species:
        counts = counts[: args.top_species]
    if not counts:
        sys.exit("No species matched. Try a lower --min-per-species or --license all.")

    os.makedirs(args.data_root, exist_ok=True)
    species = [(key, species_name(key), n) for key, n in counts]
    with open(counts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speciesKey", "species", "photo_records_on_gbif"])
        w.writerows(species)

    total_planned = sum(min(n, args.max_per_species) for _, _, n in species)
    print(f"{len(species)} species selected, about {total_planned:,} images in total.")
    for key, name, n in species[:15]:
        print(f"  {name:<40} {n:>6,} records")
    if len(species) > 15:
        print(f"  ... and {len(species) - 15} more (see {counts_path})")
    if args.dry_run:
        print("\nDry run only; nothing downloaded.")
        return

    existing = read_existing(meta_path)
    new_file = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=META_FIELDS)
        if new_file:
            writer.writeheader()

        saved_total = 0
        for si, (key, name, _) in enumerate(species, 1):
            label = safe_name(name)
            folder = os.path.join(image_root, label)
            os.makedirs(folder, exist_ok=True)
            done_ids, number = existing.get(label, (set(), 0))
            records = list(image_records(key, args, done_ids))
            saved, failed = 0, 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs = {}
                for rec in records:
                    tmp = os.path.join(folder, f".{rec['gbifID']}.part")
                    jobs[pool.submit(download, rec["url"], tmp)] = (rec, tmp)
                for fut in as_completed(jobs):
                    rec, tmp = jobs[fut]
                    if fut.result() != "ok":
                        failed += 1
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        continue
                    # number files only after a successful download, so there are no gaps
                    number += 1
                    final = os.path.join(folder, f"{label}_{number}.jpg")
                    os.replace(tmp, final)
                    writer.writerow({
                        "file": os.path.relpath(final, args.data_root),
                        "species": name, "speciesKey": key,
                        **{k: rec[k] for k in META_FIELDS if k in rec},
                    })
                    saved += 1
            mf.flush()
            saved_total += saved
            note = f", {failed} failed" if failed else ""
            print(f"[{si}/{len(species)}] {label}: {saved} new, {number} total{note}")

    print(f"\nDone: {saved_total:,} new images in {os.path.abspath(image_root)}")
    print(f"Photographer credits and licenses: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Re-run the same command to resume; saved files are kept.")
