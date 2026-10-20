# Version 1.0 (467 species)

All Kenyan species with at least 20 usable photos after license checks.

| | |
|---|---|
| Species | 467 |
| Photos | 48,513: 33,959 train / 7,277 validation / 7,277 test (stratified) |
| Photo size | shorter side 224 px |
| Model | MobileNetV2 (ImageNet weights, frozen) + softmax head |
| Training | `--cached-features`: features from each photo and its mirror image, head trained 23 epochs (best: 17), class weights for rare species |
| Command | `python3 main.py --version 1.0 --min-images 20 --cached-features --epochs 40` |

## Results on the test photos

| Metric | Value |
|---|---|
| Accuracy | 44.9% |
| Top-5 accuracy | 69.3% |
| Macro F1 | 38.4% |
| Weighted F1 | 44.3% |
| Misclassified test photos | 4,013 of 7,277 |

- 26 species reach F1 0.70 or higher, among them Grey Crowned Crane (*Balearica regulorum*),
  Saddle-billed Stork (*Ephippiorhynchus senegalensis*), Red-knobbed Coot (*Fulica cristata*) and
  Crab-plover (*Dromas ardeola*).
- Species with 25+ test photos average F1 0.51. 51 species score F1 0, mostly those with only
  3–5 test photos.
- The most frequent mix-ups are genuinely similar species: Red-billed vs Yellow-billed Oxpecker,
  Eurasian vs African Hoopoe, Greater vs Lesser Flamingo, Common vs Somali Ostrich.

Files: [metrics.json](metrics.json), [per_species_2x2.csv](per_species_2x2.csv),
[most_confused_pairs.csv](most_confused_pairs.csv), [confusion_matrix.csv](confusion_matrix.csv).
The confusion matrix and training curve charts are kept locally, not in git.
Misclassified photos are in `data/validation_data/1.0/1.0_results/misclassified/` (not in git).

## Comparison with 0.9

0.9 scored 72.2% on 50 species; 1.0 covers 467 species, so the two are not directly comparable.
On those same 50 species, 1.0 scores 54.8% (1,407 test photos). It has to choose among 467
species instead of 50, and its MobileNetV2 base was not fine-tuned.

## Next

Validation accuracy levels off near 45% by epoch 9 while training accuracy keeps rising, so
the frozen ImageNet features are the limit. Version 1.1 should fine-tune the top MobileNetV2
layers on a GPU (for example an Apple Silicon Mac):

```bash
python3 main.py --version 1.1 --min-images 20 --epochs 10 --fine-tune-epochs 10
```
