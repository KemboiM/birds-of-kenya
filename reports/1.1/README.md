# Version 1.1 (467 species, trained on boxed birds)

Same 467 species and same train/validation/test photos as 1.0, but the classifier learns
from the birds boxed in each photo (`data/annotations/`) instead of whole photos, and the top
MobileNetV2 layers are fine-tuned.

| | |
|---|---|
| Species | 467 |
| Photos | 1.0's split, minus 504 photos with no usable bird (`data/annotations/exclude.csv`) |
| Training images | 46,389 bird crops from the training photos, 9,973 from validation (up to 5 birds per photo) |
| Test images | 7,199 photos, one prediction each (the largest bird) |
| Model | MobileNetV2 (ImageNet weights) + softmax head; 10 epochs head, 10 epochs fine-tuning |
| Command | `python3 main.py --version 1.1 --from-annotations --split-from 1.0 --exclude data/annotations/exclude.csv --epochs 10 --fine-tune-epochs 10` |

## Results on the test photos

| Metric | 1.0 | 1.1 |
|---|---|---|
| Accuracy | 45.2% | **63.2%** |
| Top-5 accuracy | 69.3% | **83.0%** |
| Macro F1 | 38.4% | **56.9%** |
| Weighted F1 | 44.3% | **62.7%** |
| Misclassified test photos | 4,013 | 2,647 |

1.0's accuracy here is measured on the same 7,199 test photos as 1.1.

- 152 species reach F1 0.70 or higher (26 in 1.0). Among the best with 15+ test photos:
  Coqui Francolin (*Campocolinus coqui*) 0.97, Spur-winged Lapwing (*Vanellus spinosus*) 0.91,
  Wattled Lapwing (*Vanellus senegallus*) 0.91, Yellow-throated Sandgrouse
  (*Pterocles gutturalis*) 0.91 and Yellow-necked Spurfowl (*Pternistis leucoscepus*) 0.91.
- Species with 25+ test photos average F1 0.68 (0.51 in 1.0). 28 species score F1 0
  (51 in 1.0), mostly those with only a few test photos.
- On the 50 species of version 0.9, accuracy rose from 54.8% (1.0) to 72.5%.
- The most frequent mix-ups are close relatives: Eurasian vs African Hoopoe, Wahlberg's vs
  Tawny Eagle, Somali vs Common Ostrich, Great vs Intermediate Egret, and Grey-backed vs
  Long-tailed Fiscal.

Files: [metrics.json](metrics.json), [per_species_2x2.csv](per_species_2x2.csv),
[most_confused_pairs.csv](most_confused_pairs.csv), [confusion_matrix.csv](confusion_matrix.csv),
[confusion_matrix.png](confusion_matrix.png), [training_curves.png](training_curves.png).
Misclassified photos are in `data/validation_data/1.1/1.1_results/misclassified/` (not in git).

## Next

- Write the look-alike differences for the most-confused pairs in `species_info/`.
- More photos for the species that score F1 0; most have under 25 photos in total.
- Check the 1,894 boxes found only on the detector's second pass (`data/annotations/review.csv`).
