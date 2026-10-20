"""Identify, count and describe the birds in a photo.

    python3 identify.py photo.jpg
    python3 identify.py photos/*.jpg --version 1.0 --tile

Pipeline:
  1. YOLO finds every bird (box per bird)          -> count
  2. MobileNetV2 classifier names each bird crop    -> "12 x Marabou Stork"
  3. species_info/<Species>.json                   -> bio, sizes, look-alikes, ...
  4. box size + species size + camera focal length -> approximate distance

Outputs per photo in outputs/identify/:
  <photo>_identified.jpg   boxes numbered and labelled
  <photo>_identified.json  everything below, machine-readable
  <photo>_identified.txt   the printed report
"""

import argparse
import json
import os
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.identify.distance import describe, estimate_distance, focal_length_35mm
from src.identify.species_info import (display_name, format_sizes, load_species_info,
                                       short_name)
from src.utils.helpers import load_label_list

PROJECT = 'birds'
PADDING = 0.10            # extra margin around each box before classifying


def parse_args():
    p = argparse.ArgumentParser(description="Identify, count and describe birds in photos.")
    p.add_argument('images', nargs='+', help='photo file(s)')
    p.add_argument('--version', default='1.0', help='classifier version to use')
    p.add_argument('--detector', default='yolo11n.pt',
                   help='YOLO weights (default yolo11n.pt; yolo11s.pt/m.pt are more accurate)')
    p.add_argument('--det-conf', type=float, default=0.25, help='detector confidence threshold')
    p.add_argument('--min-confidence', type=float, default=0.5,
                   help='below this species confidence a bird is reported as unidentified')
    p.add_argument('--tile', action='store_true',
                   help='scan large photos in tiles to catch small birds in big flocks')
    p.add_argument('--out', default=os.path.join('outputs', 'identify'))
    return p.parse_args()


class SpeciesClassifier:
    def __init__(self, version):
        import keras
        path = os.path.join('model', 'model_assets', 'trained_checkpoints', version,
                            f'{PROJECT}_classify_{version}.keras')
        labels = os.path.join('model', 'configs_and_labels', f'{PROJECT}_label_list_{version}.txt')
        if not os.path.exists(path):
            raise SystemExit(f"Classifier not found: {path}\nTrain it first: python3 main.py --version {version}")
        self.model = keras.models.load_model(path)
        self.labels = load_label_list(labels)
        self.size = self.model.input_shape[1]

    def predict(self, crops):
        batch = np.stack([np.asarray(c.convert('RGB').resize((self.size, self.size)),
                                     dtype='float32') for c in crops])
        return self.model.predict(batch, verbose=0)


def crop_with_padding(img, det):
    pw, ph = det.width * PADDING, det.height * PADDING
    box = (max(0, det.x1 - pw), max(0, det.y1 - ph),
           min(img.width, det.x2 + pw), min(img.height, det.y2 + ph))
    return img.crop(tuple(int(v) for v in box))


def draw_boxes(img, birds, path):
    out = img.copy()
    d = ImageDraw.Draw(out)
    width = max(2, img.width // 400)
    try:
        font = ImageFont.truetype("Arial.ttf", max(14, img.width // 60))
    except OSError:
        font = ImageFont.load_default()
    for b in birds:
        x1, y1, x2, y2 = b['box']
        color = (40, 170, 80) if b['identified'] else (230, 150, 20)
        d.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text = f"{b['id']}. {b['name']} {b['confidence']:.0%}"
        tb = d.textbbox((x1, y1), text, font=font)
        d.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
        d.text((x1, y1), text, fill="white", font=font)
    out.save(path, quality=92)


def species_profile(info, birds):
    """Text block describing one species, shown once however many birds there are."""
    lines = [f"\n{display_name(info)}"]
    if info.get('other_names', {}).get('sw'):
        lines.append(f"  Kiswahili: {info['other_names']['sw']}")
    confs = [b['confidence'] for b in birds]
    lines.append(f"  Birds in photo: {len(birds)}   confidence {min(confs):.0%}-{max(confs):.0%}")

    def add(label, text):
        if text:
            lines.append(f"  {label}: {text}")

    add("About", info.get('summary'))
    app = info.get('appearance') or {}
    add("Colour", app.get('colors'))
    add("Wings", app.get('wings'))
    add("Appearance", app.get('description') if not (app.get('colors') or app.get('wings')) else None)
    sizes = info.get('size') or {}
    for stage in ('adult', 'juvenile', 'chick'):
        add(f"Size ({stage})", format_sizes(sizes.get(stage)))
    add("Behaviour", info.get('behaviour'))
    where = info.get('where_found') or {}
    add("Where found", where.get('habitat'))
    if where.get('kenya_regions'):
        add("Most records in Kenya", ", ".join(where['kenya_regions']))
    if 'can_fly' in info:
        add("Can it fly", "Yes" if info['can_fly'] else "No, it is flightless")
    for alike in info.get('look_alikes', [])[:3]:
        name = alike.get('common_name') or alike.get('label', '').replace('_', ' ')
        diff = alike.get('difference') or "difference not written yet"
        add("Looks similar to", f"{name}: {diff}")
    if info.get('review', {}).get('auto_generated') and not info['review'].get('verified_by'):
        lines.append("  (species notes auto-generated from open sources; not yet reviewed)")
    return "\n".join(lines)


def identify_photo(path, detector, classifier, args):
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    focal, from_exif = focal_length_35mm(path)
    dets = detector.detect(np.asarray(img), tile=args.tile)
    stem = os.path.splitext(os.path.basename(path))[0]

    birds = []
    if dets:
        probs = classifier.predict([crop_with_padding(img, d) for d in dets])
    else:
        probs = np.empty((0, len(classifier.labels)))

    for i, (det, p) in enumerate(zip(dets, probs), 1):
        top = np.argsort(p)[::-1][:3]
        label = classifier.labels[top[0]]
        info = load_species_info(label)
        identified = p[top[0]] >= args.min_confidence
        dist = estimate_distance(det, img.size, focal, (info.get('size') or {}).get('adult'))
        birds.append({
            'id': i,
            'label': label if identified else None,
            'name': short_name(info) if identified else 'Unidentified bird',
            'identified': bool(identified),
            'confidence': float(p[top[0]]),
            'top3': [{'label': classifier.labels[j], 'p': round(float(p[j]), 3)} for j in top],
            'box': [round(v) for v in (det.x1, det.y1, det.x2, det.y2)],
            'detector_score': round(det.score, 3),
            'distance': dist,
            'distance_text': describe(dist, from_exif),
        })

    by_species = defaultdict(list)
    for b in birds:
        by_species[b['label']].append(b)

    # ---- report text
    parts = []
    for label, group in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        if label is None:
            parts.append(f"{len(group)} unidentified bird{'s' if len(group) > 1 else ''}")
        else:
            info = load_species_info(label)
            parts.append(f"{len(group)} {short_name(info, len(group))}")
    lines = [f"Photo: {path}",
             f"Found {len(birds)} bird{'s' if len(birds) != 1 else ''}"
             + (": " + ", ".join(parts) if parts else
                " (no bird detected; try --tile or a lower --det-conf)")]
    for b in birds:
        alt = "" if b['identified'] else "  possible: " + ", ".join(
            f"{t['label'].replace('_', ' ')} {t['p']:.0%}" for t in b['top3'])
        lines.append(f"  #{b['id']} {b['name']} ({b['confidence']:.0%}), {b['distance_text']}{alt}")
    for label, group in sorted(by_species.items(), key=lambda kv: -len(kv[1])):
        if label is not None:
            lines.append(species_profile(load_species_info(label), group))
    report = "\n".join(lines)

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"{stem}_identified")
    draw_boxes(img, birds, base + ".jpg")
    with open(base + ".json", "w") as f:
        json.dump({'photo': path, 'focal_length_35mm': focal, 'focal_from_exif': from_exif,
                   'counts': {(k or 'unidentified'): len(v) for k, v in by_species.items()},
                   'birds': birds}, f, indent=2)
    with open(base + ".txt", "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {base}.jpg / .json / .txt\n" + "-" * 60)


def main():
    args = parse_args()
    from src.identify.detector import BirdDetector
    detector = BirdDetector(args.detector, conf=args.det_conf)
    classifier = SpeciesClassifier(args.version)
    for path in args.images:
        identify_photo(path, detector, classifier, args)


if __name__ == "__main__":
    main()
