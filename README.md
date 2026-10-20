# Birds of Kenya

A machine learning project to classify and identify bird species native to Kenya using computer vision and AI models.

## Project Goals
- Build a dataset of Kenyan bird species
- Train classification models for bird identification
- Deploy as an interactive web application

## Technologies
- Python 3.x
- TensorFlow/Keras (ML models)
- NumPy, Pandas (data processing)
- Matplotlib (visualization)

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Training images come from [GBIF](https://www.gbif.org): openly licensed field photos of
birds observed in Kenya, mostly iNaturalist research-grade observations.

```bash
# See which species have enough photos
python3 scripts/download_gbif_images.py --dry-run

# Download 50 species, up to 200 photos each (add --license commercial for CC0/CC BY only)
python3 scripts/download_gbif_images.py --top-species 50 --max-per-species 200
```

Images land in `data/gbif/images/<Species_name>/` (git-ignored).
`data/gbif/metadata.csv` records each photo's license and photographer; credit them
when you publish the model or images.

## Training
```bash
python3 main.py
```

- MobileNetV2 (ImageNet weights), softmax head, then fine-tuning of the top layers
- Stratified 70/15/15 train/validation/test split
- Outputs: `models/best_model.keras`, `models/class_names.json`,
  `results/` (accuracy, top-5, macro F1, per-species report)

Run `python3 main.py --help` for epochs, batch size and other options.

## Author
Milsort Kemboi