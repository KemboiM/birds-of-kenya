# Birds of Kenya

A bird identification tool for Kenyan species. From one photo it finds every bird,
names the species, counts them ("12 Marabou Storks, 3 Hadada Ibises"), describes each
species, and estimates how far away the birds are. It runs on a computer or, through the
web app, on Android and iOS phones.

**Status (October 2026):** version 1.0 covers 467 species with 44.9% test accuracy
(top-5 69.3%), trained without fine-tuning. Version 0.9 (50 species, 72.2%) is kept as a
baseline. Results for each version are in [reports/](reports/).

## How it works

1. **Find the birds.** YOLO11n, pretrained on COCO, draws a box around every bird.
2. **Name each bird.** A MobileNetV2 classifier trained on GBIF photos of Kenyan species
   classifies each box.
3. **Describe the species.** `species_info/<Scientific_name>.json` holds the common names,
   bio, colours, wings, sizes by life stage, behaviour, where it occurs in Kenya, whether it
   flies, and look-alike species.
4. **Estimate distance.** Approximate, from the box size, the species' known size and the
   camera focal length stored in the photo.

## Project layout

```
birds-of-kenya/
├── app/web/                    phone web app (Android and iOS)
├── data/                       on your computer only, except the train/val/test lists
│   ├── images/sorted_data/<Species>/<Species>_1.jpg ...   training photos
│   ├── images/images_from_field/                          your own photos
│   ├── gbif_metadata.csv       license and photographer for every training photo
│   ├── training_data/trainvals/{train,val,test}_birds_<version>.txt
│   └── validation_data/<version>/<version>_results/       full evaluation incl. misclassified/
├── docs/mobile.md              running the models in Android and iOS apps
├── model/
│   ├── configs_and_labels/     label lists and training settings per version
│   └── model_assets/           trained_checkpoints/<version>/, tflite_models/
├── protocols/                  naming and sourcing, annotation, field testing
├── reports/<version>/          results committed to git (metrics, confusion matrix, ...)
├── scripts/                    download data, build species info, export models
├── species_info/               species knowledge base, one JSON per species
├── src/                        data loading, training, evaluation, identification
├── main.py                     train a version
└── identify.py                 identify, count and describe birds in photos
```

Naming rules: [protocols/image_naming_and_sourcing.md](protocols/image_naming_and_sourcing.md).

## Setup

Python 3.10–3.12.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Download training photos

```bash
python3 scripts/download_gbif_images.py --dry-run   # species and photo counts only
python3 scripts/download_gbif_images.py             # every species with 20+ photos, up to 200 each
```

Each photo's own license is checked. Only CC0, CC BY, CC BY-SA, CC BY-NC and CC BY-NC-SA
photos are kept (`--license commercial` keeps only the first three). No-derivatives and
all-rights-reserved photos, and sound spectrograms listed as images, are skipped.
Re-running resumes where it stopped.

## 2. Train

```bash
python3 main.py --version 1.0
```

Training writes the model to `model/model_assets/trained_checkpoints/1.0/` and the full
evaluation to `data/validation_data/1.0/1.0_results/`:

| Output | Contents |
|---|---|
| `metrics.json` | accuracy, top-5 accuracy, macro and weighted F1 |
| `confusion_matrix.csv`, `confusion_matrix_normalized.csv` | full N x N matrix, rows = true species |
| `confusion_matrix.png` | the matrix, or its 30 most-confused species when N is large |
| `per_species_2x2.csv` | TP, FN, FP, TN, precision, recall and F1 per species, hardest first |
| `most_confused_pairs.csv` | which species are mistaken for which |
| `misclassified/<True>__predicted__<Pred>/` | every wrong test photo, confidence in the file name |
| `misclassified.csv` | the same list with the top 3 guesses and GBIF source links |
| `training_curves.png` | accuracy and loss per epoch |

The small files are also copied to `reports/1.0/` for git. Add `--copy-correct` to also
save correctly classified photos. Use a new `--version` for every run.

On a computer without a GPU, add `--cached-features`: MobileNetV2 features are computed
once per photo (plus a mirrored copy) and only the classifier head is trained, which takes
minutes instead of hours. It skips fine-tuning, so a GPU run without the flag usually
scores a few points higher.

## 3. Build the species knowledge base

```bash
python3 scripts/build_species_info.py \
    --confusion data/validation_data/1.0/1.0_results/most_confused_pairs.csv
```

Fills each species file from GBIF and Wikipedia and adds look-alikes from the confusion
matrix. Check sizes against a field guide (distance depends on them), write how to tell
each look-alike apart, then set `review.verified_by`. See [species_info/README.md](species_info/README.md).

## 4. Identify birds

```bash
python3 identify.py photo.jpg --version 1.0
python3 identify.py flock.jpg --version 1.0 --tile     # large flocks, distant birds
```

Saves `outputs/identify/<photo>_identified.jpg` (numbered boxes), `.json` and `.txt`.

## 5. Phones

- **Web app, Android and iOS:** `python3 scripts/export_web.py --version 1.0`, then publish
  `app/web/` ([app/web/README.md](app/web/README.md)).
- **Native apps:** `python3 scripts/export_tflite.py --version 1.0 --detector` ([docs/mobile.md](docs/mobile.md)).

Exports need `pip install -r requirements-export.txt`.

## Limits

- Distance is an estimate, often off by 30–50%. Pose, partly hidden birds, digital zoom
  and cropped photos all add error, and it needs the species' size in `species_info/`.
- Dense flocks such as flamingos are undercounted; `--tile` helps.
- A bird of a species the model was not trained on gets the closest wrong name.
- Species notes are auto-generated until reviewed, and reports say so.

## Licenses

| Part | License |
|---|---|
| Code in this repository | AGPL-3.0 ([LICENSE](LICENSE)), because it builds on Ultralytics YOLO, which is AGPL-3.0 |
| Training photos (not in the repository) | each photo's own Creative Commons license, listed in `data/gbif_metadata.csv`; photographers must be credited when photos are shown |
| Trained classifier | trained on CC BY-NC photos, so for non-commercial use only |
| YOLO11n detector weights | AGPL-3.0 (Ultralytics); commercial closed-source use needs an Ultralytics license |
| MobileNetV2 starting weights | Keras Applications (Apache 2.0), trained on ImageNet |
| `species_info/` text | Wikipedia text is CC BY-SA 4.0; GBIF data as stated per dataset; sources linked in each file |
| Phone app runtime | ONNX Runtime Web, MIT |

## Credits

Training photos: [GBIF](https://www.gbif.org) contributors, mostly
[iNaturalist](https://www.inaturalist.org) observers. Detector:
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

## Author

Milsort Kemboi
