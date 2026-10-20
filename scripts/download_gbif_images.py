#!/usr/bin/env python3
"""
Download openly licensed photos of Kenyan birds from GBIF into one folder
per species, ready for an image classifier (e.g. MobileNetV2 transfer learning).

Source: GBIF occurrence API (country=KE, class Aves, records with images).
Most photos come from iNaturalist research-grade observations.

Output layout:
    <out>/images/<Species_name>/<gbifID>.jpg
    <out>/metadata.csv      one row per image: species, license, photographer, URL
    <out>/species_counts.csv

Only the Python standard library is needed (Python 3.8+).

Examples:
    # See which species have enough photos, download nothing
    python3 scripts/download_gbif_images.py --dry-run

    # 50 species with the most photos, up to 200 images each, commercial-safe licenses only
    python3 scripts/download_gbif_images.py --top-species 50 --max-per-species 200 --license commercial

    # Every species with at least 30 photos, all open licenses
    python3 scripts/download_gbif_images.py --min-per-species 30 --max-per-species 300
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
USER_AGENT = "kenya-bird-images/1.0 (research; birds-of-kenya classifier)"

LICENSES = {
    # GBIF license filter values
    "commercial": ["CC0_1_0", "CC_BY_4_0"],                 # safe for any use
    "all": ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"],        # includes non-commercial
}


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
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "exists"
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                ctype = r.headers.get("Content-Type", "")
                if "image" not in ctype:
                    return f"skipped (content-type {ctype or 'unknown'})"
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
            os.replace(tmp, dest)
            return "ok"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return f"failed ({e})"
            time.sleep(2 ** attempt)


def build_url(path, params):
    # params may contain lists -> repeated keys (GBIF ORs repeated filters)
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
    params = dict(base_filters(args), limit=0, facet="speciesKey", facetLimit=3000)
    data = get_json(build_url("/occurrence/search", params))
    facets = data.get("facets") or []
    counts = facets[0]["counts"] if facets else []
    return [(int(c["name"]), int(c["count"])) for c in counts]


def species_name(key, cache={}):
    if key not in cache:
        sp = get_json(f"{API}/species/{key}")
        cache[key] = sp.get("canonicalName") or sp.get("scientificName") or str(key)
    return cache[key]


def safe_dir(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")


def resize_url(url, size):
    """iNaturalist hosts several sizes; 'medium' (~500px) is plenty for 224px models."""
    if size == "original" or "inaturalist" not in url:
        return url
    return re.sub(r"/(original|large|medium|small|square)\.", f"/{size}.", url)


def image_records(species_key, args):
    """Yield one record per image for a species, up to --max-per-species."""
    taken, offset = 0, 0
    params = dict(base_filters(args), speciesKey=species_key, limit=PAGE)
    while taken < args.max_per_species and offset < GBIF_OFFSET_CAP:
        params["offset"] = offset
        data = get_json(build_url("/occurrence/search", params))
        for occ in data.get("results", []):
            stills = [m for m in occ.get("media", [])
                      if m.get("type") == "StillImage" and m.get("identifier")]
            # One photo per observation by default: extra photos of the same bird
            # are near-duplicates and leak between train/val splits.
            for i, m in enumerate(stills if args.all_media else stills[:1]):
                yield {
                    "gbifID": occ.get("gbifID") or occ.get("key"),
                    "index": i,
                    "url": resize_url(m["identifier"], args.size),
                    "license": m.get("license") or occ.get("license", ""),
                    "rightsHolder": m.get("rightsHolder") or m.get("creator") or "",
                    "eventDate": occ.get("eventDate", ""),
                    "lat": occ.get("decimalLatitude", ""),
                    "lon": occ.get("decimalLongitude", ""),
                    "dataset": occ.get("datasetName", ""),
                    "occurrence_url": f"https://www.gbif.org/occurrence/{occ.get('key')}",
                }
                taken += 1
                if taken >= args.max_per_species:
                    return
        if data.get("endOfRecords", True):
            return
        offset += PAGE


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/gbif", help="output folder (default: data/gbif)")
    p.add_argument("--top-species", type=int, default=0,
                   help="keep only the N species with the most photos (0 = no limit)")
    p.add_argument("--min-per-species", type=int, default=30,
                   help="skip species with fewer photo records than this (default: 30)")
    p.add_argument("--max-per-species", type=int, default=200,
                   help="cap images per species to keep classes balanced (default: 200)")
    p.add_argument("--license", choices=LICENSES, default="all",
                   help="'commercial' = CC0 and CC BY only; 'all' adds CC BY-NC (default: all)")
    p.add_argument("--size", choices=["medium", "large", "original"], default="medium",
                   help="iNaturalist image size to fetch (default: medium, ~500px)")
    p.add_argument("--all-media", action="store_true",
                   help="take every photo per observation, not just the first")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (default: 8)")
    p.add_argument("--dry-run", action="store_true",
                   help="only list species and photo counts; download nothing")
    args = p.parse_args()

    print("Querying GBIF for Kenyan bird species with photos...")
    counts = species_counts(args)
    counts = [(k, n) for k, n in counts if n >= args.min_per_species]
    counts.sort(key=lambda kn: kn[1], reverse=True)
    if args.top_species:
        counts = counts[: args.top_species]
    if not counts:
        sys.exit("No species matched. Try a lower --min-per-species or --license all.")

    os.makedirs(args.out, exist_ok=True)
    species = []
    for key, n in counts:
        species.append((key, species_name(key), n))

    with open(os.path.join(args.out, "species_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speciesKey", "species", "photo_records_on_gbif"])
        w.writerows(species)

    total_planned = sum(min(n, args.max_per_species) for _, _, n in species)
    print(f"{len(species)} species selected, about {total_planned:,} images to fetch.")
    for key, name, n in species[:15]:
        print(f"  {name:<40} {n:>6,} records")
    if len(species) > 15:
        print(f"  ... and {len(species) - 15} more (see species_counts.csv)")
    if args.dry_run:
        print(f"\nDry run only. Species list saved to {args.out}/species_counts.csv")
        return

    meta_path = os.path.join(args.out, "metadata.csv")
    fields = ["species", "speciesKey", "file", "gbifID", "url", "license",
              "rightsHolder", "eventDate", "lat", "lon", "dataset", "occurrence_url", "status"]
    new_file = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fields)
        if new_file:
            writer.writeheader()

        ok_total = 0
        for si, (key, name, _) in enumerate(species, 1):
            folder = os.path.join(args.out, "images", safe_dir(name))
            os.makedirs(folder, exist_ok=True)
            records = list(image_records(key, args))
            jobs = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for rec in records:
                    suffix = f"_{rec['index']}" if rec["index"] else ""
                    path = os.path.join(folder, f"{rec['gbifID']}{suffix}.jpg")
                    jobs[pool.submit(download, rec["url"], path)] = (rec, path)
                ok = 0
                for fut in as_completed(jobs):
                    rec, path = jobs[fut]
                    status = fut.result()
                    if status in ("ok", "exists"):
                        ok += 1
                    writer.writerow({
                        "species": name, "speciesKey": key,
                        "file": os.path.relpath(path, args.out),
                        "status": status,
                        **{k: rec[k] for k in fields if k in rec},
                    })
            mf.flush()
            ok_total += ok
            print(f"[{si}/{len(species)}] {name}: {ok}/{len(records)} images")

    print(f"\nDone: {ok_total:,} images in {os.path.abspath(os.path.join(args.out, 'images'))}")
    print("Credit photographers per metadata.csv (rightsHolder + license) when you publish.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Re-run the same command to resume; finished files are skipped.")
