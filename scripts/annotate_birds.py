#!/usr/bin/env python3
"""
Draw a box around every bird in every training photo and save Pascal VOC XML
annotations (the LabelImg format), labelled with the folder's scientific name.

    data/images/sorted_data/<Species>/<Species>_N.jpg
 -> data/annotations/<Species>/<Species>_N.xml
    data/annotations/annotation_summary.csv     one row per photo, with a status
    data/annotations/review.csv                 photos and boxes that need a person to check

Works folder by folder, photo by photo, and resumes where it stopped.

Two detector passes:
  1. a fast model at normal confidence for every photo;
  2. for photos where pass 1 finds nothing (birds in flight against the sky, birds
     blending into rock, bark or reeds, birds on nests, small birds in a flock),
     a larger model at low confidence with test-time augmentation and a bigger
     input size.

Statuses in annotation_summary.csv:
  ok                       boxes from pass 1
  found_on_second_pass     boxes only from pass 2: check them, they are less certain
  no_bird_found            no box; may be a non-bird image (e.g. a sound
                           spectrogram) or a very hidden bird
  other_species_suspected  a box that the species classifier thinks is another
                           species (mixed flocks, prey, a different bird in the frame)

Every box is labelled with the folder's species. Boxes flagged in review.csv
may show another species and should be checked in LabelImg.

Example:
    python3 scripts/annotate_birds.py
    python3 scripts/annotate_birds.py --species Acryllium_vulturinum Agapornis_fischeri
"""

import argparse
import csv
import os
import sys
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageOps

BIRD = 14            # COCO class "bird"
SUMMARY_FIELDS = ["image", "species", "width", "height", "boxes", "max_score",
                  "detector_pass", "status", "note"]
REVIEW_FIELDS = ["image", "species", "status", "box", "score", "classifier_top",
                 "classifier_top_prob", "folder_species_prob", "note"]


def load_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im.convert("RGB")


def voc_xml(label, filename, rel_path, w, h, boxes):
    objs = []
    for (x1, y1, x2, y2), _ in boxes:
        truncated = int(x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3)
        objs.append(f"""\t<object>
\t\t<name>{escape(label)}</name>
\t\t<pose>Unspecified</pose>
\t\t<truncated>{truncated}</truncated>
\t\t<difficult>0</difficult>
\t\t<bndbox>
\t\t\t<xmin>{x1}</xmin>
\t\t\t<ymin>{y1}</ymin>
\t\t\t<xmax>{x2}</xmax>
\t\t\t<ymax>{y2}</ymax>
\t\t</bndbox>
\t</object>""")
    return f"""<annotation>
\t<folder>{escape(label)}</folder>
\t<filename>{escape(filename)}</filename>
\t<path>{escape(rel_path)}</path>
\t<source>
\t\t<database>GBIF</database>
\t</source>
\t<size>
\t\t<width>{w}</width>
\t\t<height>{h}</height>
\t\t<depth>3</depth>
\t</size>
\t<segmented>0</segmented>
{chr(10).join(objs)}
</annotation>
""".replace("\n\n</annotation>", "\n</annotation>")


class Detector:
    def __init__(self, weights, conf, imgsz, augment=False, device="auto"):
        import torch
        import torchvision  # noqa: F401  (load PyTorch fully before TensorFlow)
        from ultralytics import YOLO
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.augment, self.device = conf, imgsz, augment, device
        self.model.predict(np.zeros((64, 64, 3), np.uint8), device=device, verbose=False)

    def __call__(self, images):
        results = self.model.predict(images, conf=self.conf, imgsz=self.imgsz, classes=[BIRD],
                                     augment=self.augment, max_det=300, device=self.device,
                                     verbose=False)
        out = []
        for r in results:
            boxes = []
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                boxes.append(((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(s)))
            boxes.sort(key=lambda b: -(b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
            out.append(boxes)
        return out


class SpeciesCheck:
    """Optional: flags boxes the trained classifier confidently assigns to another species."""

    def __init__(self, version):
        import keras
        base = os.path.join("model", "model_assets", "trained_checkpoints", version)
        self.model = keras.models.load_model(os.path.join(base, f"birds_classify_{version}.keras"))
        with open(os.path.join("model", "configs_and_labels", f"birds_label_list_{version}.txt")) as f:
            self.labels = [l.strip() for l in f if l.strip()]
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.size = self.model.input_shape[1]

    def __call__(self, img, boxes):
        if not boxes:
            return []
        crops = []
        for (x1, y1, x2, y2), _ in boxes:
            pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            c = img.crop((max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph)))
            crops.append(np.asarray(c.resize((self.size, self.size)), dtype="float32"))
        return self.model.predict(np.stack(crops), verbose=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--species", nargs="*", help="only these species folders")
    p.add_argument("--weights", default="yolo11s.pt", help="pass 1 detector")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--second-weights", default="yolo11m.pt", help="pass 2 detector for misses")
    p.add_argument("--second-conf", type=float, default=0.05)
    p.add_argument("--min-box", type=float, default=0.0015,
                   help="drop boxes smaller than this share of the photo (noise)")
    p.add_argument("--check-version", default="",
                   help="classifier version used to flag boxes of other species, e.g. 1.1 (off by default)")
    p.add_argument("--device", default="auto",
                   help="auto, cpu, mps (Apple Silicon GPU) or a CUDA index such as 0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--redo", action="store_true", help="re-annotate photos that already have XML")
    args = p.parse_args()

    img_root = os.path.join(args.data_root, "images", "sorted_data")
    ann_root = os.path.join(args.data_root, "annotations")
    os.makedirs(ann_root, exist_ok=True)
    folders = args.species or sorted(d for d in os.listdir(img_root)
                                     if os.path.isdir(os.path.join(img_root, d)))

    det1 = Detector(args.weights, args.conf, 640, device=args.device)
    det2 = Detector(args.second_weights, args.second_conf, 960, augment=True, device=args.device)
    check = SpeciesCheck(args.check_version) if args.check_version else None

    summary_path = os.path.join(ann_root, "annotation_summary.csv")
    review_path = os.path.join(ann_root, "review.csv")
    new_summary = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as sf, open(review_path, "a", newline="") as rf:
        sw, rw = csv.DictWriter(sf, SUMMARY_FIELDS), csv.DictWriter(rf, REVIEW_FIELDS)
        if new_summary:
            sw.writeheader()
            rw.writeheader()
        for fi, label in enumerate(folders, 1):
            files = sorted((f for f in os.listdir(os.path.join(img_root, label)) if not f.startswith(".")),
                           key=lambda f: (len(f), f))
            out_dir = os.path.join(ann_root, label)
            os.makedirs(out_dir, exist_ok=True)
            todo = [f for f in files if args.redo or not os.path.exists(
                os.path.join(out_dir, os.path.splitext(f)[0] + ".xml"))]
            counts = {}
            for i in range(0, len(todo), args.batch):
                names = todo[i:i + args.batch]
                imgs = [load_rgb(os.path.join(img_root, label, n)) for n in names]
                first = det1(imgs)
                for name, img, boxes in zip(names, imgs, first):
                    w, h = img.size
                    passes = "1"
                    boxes = [b for b in boxes if (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]) >= args.min_box * w * h]
                    if not boxes:
                        boxes = [b for b in det2([img])[0]
                                 if (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]) >= args.min_box * w * h]
                        passes = "2" if boxes else ""
                    status = "ok" if passes == "1" else ("found_on_second_pass" if passes == "2" else "no_bird_found")
                    rel = os.path.join("images", "sorted_data", label, name)
                    stem = os.path.splitext(name)[0]
                    with open(os.path.join(out_dir, stem + ".xml"), "w") as f:
                        f.write(voc_xml(label, name, rel, w, h, boxes))

                    note = ""
                    if check and boxes and label in check.index:
                        probs = check(img, boxes)
                        own = check.index[label]
                        for (box, score), pr in zip(boxes, probs):
                            top = int(pr.argmax())
                            if top != own and pr[top] >= 0.6 and pr[own] < 0.1:
                                status = "other_species_suspected"
                                rw.writerow({"image": rel, "species": label, "status": status,
                                             "box": " ".join(map(str, box)), "score": f"{score:.3f}",
                                             "classifier_top": check.labels[top],
                                             "classifier_top_prob": f"{pr[top]:.3f}",
                                             "folder_species_prob": f"{pr[own]:.3f}", "note": ""})
                    if status in ("found_on_second_pass", "no_bird_found"):
                        rw.writerow({"image": rel, "species": label, "status": status,
                                     "box": "; ".join(" ".join(map(str, b[0])) for b in boxes),
                                     "score": " ".join(f"{b[1]:.3f}" for b in boxes),
                                     "classifier_top": "", "classifier_top_prob": "",
                                     "folder_species_prob": "", "note": note})
                    sw.writerow({"image": rel, "species": label, "width": w, "height": h,
                                 "boxes": len(boxes), "max_score": f"{max((b[1] for b in boxes), default=0):.3f}",
                                 "detector_pass": passes, "status": status, "note": note})
                    counts[status] = counts.get(status, 0) + 1
                sf.flush()
                rf.flush()
            done = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "already done"
            print(f"[{fi}/{len(folders)}] {label}: {len(todo)} photos ({done})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run again to continue; finished photos are skipped.")
