#!/usr/bin/env python3
"""
Build species_info/<Scientific_name>.json, the identification tool's knowledge base,
for every species in the label list.

Filled automatically from open sources:
    GBIF       English and Kiswahili common names, family, Kenyan regions with most records
    Wikipedia  summary (bio), appearance, colours, wings, behaviour, habitat, breeding notes,
               adult length / wingspan / weight parsed from the text  (CC BY-SA 4.0)
    model      look-alike species from the confusion matrix (--confusion)

Left for you to write or check (shown as "not recorded yet" in reports):
    juvenile and chick sizes, and the written difference for each look-alike.

Every generated file is marked review.auto_generated = true. After you check a
file, set review.verified_by to your name: later runs will then leave it alone,
apart from adding new look-alikes. Parsed sizes are a best guess; check them,
because distance estimates depend on them.

Examples:
    python3 scripts/build_species_info.py
    python3 scripts/build_species_info.py --confusion data/validation_data/1.0/1.0_results/most_confused_pairs.csv
    python3 scripts/build_species_info.py --species Haliaeetus_vocifer --force
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

GBIF = "https://api.gbif.org/v1"
WIKI = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "birds-of-kenya/1.0 (species identification research tool)"

# Kenyan birds that cannot fly
FLIGHTLESS = {"Struthio_camelus", "Struthio_molybdophanes"}

SECTION_KEYS = {
    "appearance": ("description", "identification", "appearance", "plumage", "morphology"),
    "behaviour": ("behaviour", "behavior", "ecology", "diet", "feeding", "voice", "calls"),
    "habitat": ("distribution", "habitat", "range"),
    "breeding": ("breeding", "reproduction", "nesting"),
}
COLOR_WORDS = ("black", "white", "grey", "gray", "brown", "chestnut", "rufous", "red",
               "orange", "yellow", "green", "blue", "purple", "violet", "pink", "buff",
               "cream", "olive", "iridescent", "glossy", "plumage", "colour", "color")


# ---------------------------------------------------------------- HTTP

def get_json(url, params=None, retries=3):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)


# ---------------------------------------------------------------- text helpers

def sentences(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def first_sentences(text, n=3, max_chars=600):
    out = ""
    for s in sentences(text)[:n]:
        if len(out) + len(s) > max_chars and out:
            break
        out = f"{out} {s}".strip()
    return out or None


def pick(text, words, n=2, exclude=()):
    hits = [s for s in sentences(text)
            if any(w in s.lower() for w in words) and not any(x in s.lower() for x in exclude)]
    return " ".join(hits[:n]) or None


def split_sections(extract):
    """Wikipedia plain text with '== Heading ==' markers -> (lead, {heading: text})."""
    parts = re.split(r"\n\s*(={2,})\s*(.+?)\s*\1\s*\n", "\n" + (extract or "") + "\n")
    lead, sections = parts[0].strip(), {}
    for i in range(1, len(parts) - 2, 3):
        sections[parts[i + 1].strip().lower()] = parts[i + 2].strip()
    return lead, sections


def section_text(sections, keys):
    return "\n".join(t for h, t in sections.items() if any(k in h for k in keys) and t) or ""


NUM = r"(\d+(?:[.,]\d+)?)"
RANGE = rf"{NUM}\s*(?:(?:–|-|—|to)\s*{NUM})?\s*(mm|cm|m|kg|g)\b"
TO_CM = {"mm": 0.1, "cm": 1.0, "m": 100.0}
TO_G = {"g": 1.0, "kg": 1000.0}


def _values(sentence, units):
    found = []
    for a, b, unit in re.findall(RANGE, sentence):
        if unit not in units:
            continue
        lo = float(a.replace(",", ""))
        hi = float(b.replace(",", "")) if b else lo
        found.append([round(lo * units[unit], 1), round(hi * units[unit], 1)])
    return found


def _span(values):
    """Several ranges in one sentence (e.g. males and females) -> one overall range."""
    return [min(v[0] for v in values), max(v[1] for v in values)] if values else None


def parse_sizes(text):
    """Best-effort adult length / wingspan / mass from Wikipedia sentences."""
    size = {}
    for s in sentences(text):
        low = s.lower()
        if any(w in low for w in ("juvenile", "chick", "egg", "nestling", "immature")):
            continue
        key = "wingspan" if "wingspan" in low else ("wing span" if "wing span" in low else None)
        before = s[:low.index(key)] if key else s
        if key and "wingspan_cm" not in size:
            # only numbers after the word "wingspan" belong to it
            v = [x for x in _values(s[low.index(key):], TO_CM) if x[0] >= 10]
            if v:
                size["wingspan_cm"] = v[0]
        if ("long" in low or "length" in low) and "length_cm" not in size:
            v = [x for x in _values(before, TO_CM) if 5 <= x[0] <= 300]
            if v:
                size["length_cm"] = _span(v)
        if any(w in low for w in ("weigh", "mass")) and "mass_g" not in size:
            v = _values(s, TO_G)
            if v:
                size["mass_g"] = _span(v)
    return size or None


# ---------------------------------------------------------------- sources

def gbif_facts(scientific):
    match = get_json(f"{GBIF}/species/match", {"name": scientific, "class": "Aves"}) or {}
    key = match.get("usageKey")
    facts = {"family": match.get("family"), "gbif_key": key, "names": {}, "regions": [], "records": None}
    if not key:
        return facts
    names = get_json(f"{GBIF}/species/{key}/vernacularNames", {"limit": 300}) or {}
    for row in names.get("results", []):
        lang = row.get("language")
        if lang in ("eng", "swa") and lang not in facts["names"]:
            facts["names"][lang] = row.get("vernacularName")
    occ = get_json(f"{GBIF}/occurrence/search", {
        "country": "KE", "taxonKey": key, "limit": 0,
        "facet": "stateProvince", "facetLimit": 6}) or {}
    facts["records"] = occ.get("count")
    for f in occ.get("facets", []):
        facts["regions"] = [c["name"] for c in f.get("counts", []) if c.get("name")]
    return facts


def wikipedia(scientific):
    data = get_json(WIKI, {"action": "query", "prop": "extracts|info", "explaintext": 1,
                           "exsectionformat": "wiki", "inprop": "url", "redirects": 1,
                           "titles": scientific, "format": "json"}) or {}
    pages = (data.get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), {})
    if not page or "missing" in page or not page.get("extract"):
        return None
    return {"title": page.get("title"), "url": page.get("fullurl"), "extract": page["extract"]}


# ---------------------------------------------------------------- build

def load_confusions(path):
    pairs = {}
    if not path:
        return pairs
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for a, b in ((row["true_species"], row["predicted_as"]),
                         (row["predicted_as"], row["true_species"])):
                pairs.setdefault(a, {}).setdefault(b, 0)
                pairs[a][b] += int(row["times"])
    return pairs


def common_name(facts, wiki, scientific):
    """GBIF English name; else the Wikipedia title when it isn't just the Latin name."""
    if facts["names"].get("eng"):
        return facts["names"]["eng"]
    title = (wiki or {}).get("title") or ""
    return title if title and title.lower() != scientific.lower() else None


def build_entry(label, facts, wiki):
    scientific = label.replace("_", " ")
    lead, sections = split_sections(wiki["extract"]) if wiki else ("", {})
    appearance = section_text(sections, SECTION_KEYS["appearance"])
    behaviour = section_text(sections, SECTION_KEYS["behaviour"])
    habitat = section_text(sections, SECTION_KEYS["habitat"])
    breeding = section_text(sections, SECTION_KEYS["breeding"])
    body = f"{lead}\n{appearance}"

    juvenile = pick(appearance, ("juvenile", "immature", "young bird"), n=2)
    chick = pick(breeding, ("chick", "nestling", "hatchling", "hatch", "fledg"), n=2)

    sources = []
    if wiki:
        sources.append({"name": "Wikipedia", "title": wiki["title"], "url": wiki["url"],
                        "license": "CC BY-SA 4.0"})
    if facts.get("gbif_key"):
        sources.append({"name": "GBIF", "url": f"https://www.gbif.org/species/{facts['gbif_key']}"})

    return {
        "label": label,
        "scientific_name": scientific,
        "common_name": common_name(facts, wiki, scientific),
        "other_names": {"sw": facts["names"].get("swa")} if facts["names"].get("swa") else {},
        "family": facts.get("family"),
        "summary": first_sentences(lead, 3),
        "appearance": {
            "description": first_sentences(appearance, 3),
            "colors": pick(body, COLOR_WORDS, n=2, exclude=("juvenile", "immature", "egg")),
            "wings": pick(body, ("wing",), n=2, exclude=("wingspan",)),
        },
        "size": {
            "adult": parse_sizes(body),
            "juvenile": {"notes": juvenile} if juvenile else None,
            "chick": {"notes": chick} if chick else None,
        },
        "behaviour": first_sentences(behaviour, 3),
        "can_fly": label not in FLIGHTLESS,
        "where_found": {
            "habitat": first_sentences(habitat, 2),
            "kenya_regions": facts.get("regions") or [],
            "gbif_records_in_kenya": facts.get("records"),
        },
        "look_alikes": [],
        "sources": sources,
        "review": {"auto_generated": True, "verified_by": None,
                   "note": "Check sizes and text against a field guide, then set verified_by."},
    }


def merge_look_alikes(entry, label, confusions, names):
    known = {a["label"]: a for a in entry.get("look_alikes", [])}
    for other, times in sorted(confusions.get(label, {}).items(), key=lambda kv: -kv[1]):
        item = known.setdefault(other, {"label": other, "common_name": names.get(other),
                                        "difference": None})
        item["times_confused"] = times
    entry["look_alikes"] = sorted(known.values(), key=lambda a: -a.get("times_confused", 0))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="model/configs_and_labels/birds_label_list.txt")
    p.add_argument("--out", default="species_info")
    p.add_argument("--confusion", help="most_confused_pairs.csv from a training run")
    p.add_argument("--species", nargs="*", help="only these labels")
    p.add_argument("--force", action="store_true", help="rebuild existing files that are not yet verified")
    p.add_argument("--delay", type=float, default=0.3, help="seconds between species (be polite)")
    args = p.parse_args()

    if args.species:
        labels = args.species
    else:
        if not os.path.exists(args.labels):
            sys.exit(f"Label list not found: {args.labels}. Train a model first, or pass --species.")
        with open(args.labels) as f:
            labels = [l.strip() for l in f if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    confusions = load_confusions(args.confusion)

    entries, made, kept, failed = {}, 0, 0, []
    for i, label in enumerate(labels, 1):
        path = os.path.join(args.out, f"{label}.json")
        existing = None
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        verified = bool(existing and existing.get("review", {}).get("verified_by"))
        if existing and (verified or not args.force):
            entries[label] = existing
            kept += 1
            continue
        try:
            facts = gbif_facts(label.replace("_", " "))
            wiki = wikipedia(label.replace("_", " "))
            entry = build_entry(label, facts, wiki)
            if existing:
                entry["look_alikes"] = existing.get("look_alikes", [])
            entries[label] = entry
            made += 1
            status = "ok" if wiki else "no Wikipedia page"
            print(f"[{i}/{len(labels)}] {label}: {entry.get('common_name') or '-'} ({status})")
        except Exception as e:     # keep going; one bad species shouldn't stop the run
            failed.append(label)
            print(f"[{i}/{len(labels)}] {label}: FAILED ({e})")
        time.sleep(args.delay)

    names = {k: v.get("common_name") for k, v in entries.items()}
    for label, entry in entries.items():
        merge_look_alikes(entry, label, confusions, names)
        with open(os.path.join(args.out, f"{label}.json"), "w") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    missing_size = [k for k, v in entries.items() if not (v.get("size") or {}).get("adult")]
    print(f"\n{made} built, {kept} kept (already present), {len(failed)} failed.")
    if missing_size:
        print(f"{len(missing_size)} species have no adult size, so no distance estimate: "
              f"{', '.join(missing_size[:10])}{' ...' if len(missing_size) > 10 else ''}")
    if failed:
        print("Re-run to retry failed species: " + " ".join(failed))


if __name__ == "__main__":
    main()
