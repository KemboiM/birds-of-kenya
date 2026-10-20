"""Read species_info/<Scientific_name>.json (the identification tool's knowledge base)."""

import json
import os

SPECIES_INFO_DIR = "species_info"


def load_species_info(label, folder=SPECIES_INFO_DIR):
    path = os.path.join(folder, f"{label}.json")
    if not os.path.exists(path):
        return {"label": label, "scientific_name": label.replace("_", " ")}
    with open(path) as f:
        return json.load(f)


def display_name(info):
    common = info.get("common_name")
    sci = info.get("scientific_name") or info.get("label", "").replace("_", " ")
    return f"{common} ({sci})" if common else sci


def short_name(info, count=1):
    name = info.get("common_name") or info.get("scientific_name") or info.get("label")
    if count == 1 or not info.get("common_name"):
        return name
    return name if name.endswith("s") else name + "s"


def _fmt_range(values, unit):
    if not values:
        return None
    if isinstance(values, (int, float)):
        return f"{values:g} {unit}"
    lo, hi = min(values), max(values)
    return f"{lo:g} {unit}" if lo == hi else f"{lo:g}-{hi:g} {unit}"


def format_sizes(stage):
    """{'length_cm': [63, 75], 'wingspan_cm': [175, 240], 'mass_g': [2000, 3600]} -> text."""
    if not stage:
        return None
    parts = []
    for key, label, unit in (("length_cm", "length", "cm"), ("wingspan_cm", "wingspan", "cm"),
                             ("mass_g", "weight", "g")):
        text = _fmt_range(stage.get(key), unit)
        if text:
            parts.append(f"{label} {text}")
    if stage.get("notes"):
        parts.append(stage["notes"])
    return "; ".join(parts) or None
