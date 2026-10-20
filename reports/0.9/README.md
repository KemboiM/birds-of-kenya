# Version 0.9 (baseline, 50 species)

The first multi-species model, trained on 20 October 2026 with the code from the first
GBIF commit (before the folder restructure). It is kept as a baseline and powers the
current phone test app.

| | |
|---|---|
| Species | 50 (the 50 Kenyan species with the most GBIF photos) |
| Images | up to 200 per species; 70 / 15 / 15 train / validation / test split |
| Test images | 1,500 (30 per species) |
| Model | MobileNetV2 (ImageNet weights) + softmax head, fine-tuned; 224 x 224 input |
| Training | 20 epochs in total, batch size 32 |

## Results on the test images

| Metric | Value |
|---|---|
| Accuracy | 72.2% |
| Top-5 accuracy | 90.1% |
| Macro precision | 73.4% |
| Macro recall | 72.2% |
| Macro F1 | 72.3% |

Hardest species (lowest F1): Colius striatus 0.47, Ardea cinerea 0.51, Buteo augur 0.51,
Polemaetus bellicosus 0.52, Pycnonotus barbatus 0.57, Bubulcus ibis 0.59.

Easiest: Buphagus erythrorhynchus 0.90, Struthio camelus 0.89, Actophilornis africanus 0.88,
Pternistis leucoscepus 0.87, Coracias caudatus 0.87.

Per-species precision, recall and F1: [per_species.json](per_species.json).

## Known issues

- No confusion matrix or misclassified-image folders: those outputs were added after
  this run. Version 1.0 produces them.
- The training images for this run were downloaded before per-photo license checks were
  added, so they included some no-derivatives photos and xeno-canto sound spectrograms.
  Those have since been removed from the dataset.
- Trained on whole photos, while the app classifies cropped birds, which costs some
  accuracy on field photos.
