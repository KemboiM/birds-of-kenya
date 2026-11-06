# Changelog

What changed in each model version and in the tool around it. Full results for every
version are in [reports/](reports/).

## Since 2.1 (no new model)

- Phone app renamed **Birdiie**, with a crowned-crane icon. It can now be installed from
  Chrome (Android) or Safari (iPhone) and works offline after the first visit.
- Three QR codes in the README: install on Android, install on iPhone, or just open it.
- The app shows English common names, the scientific name and Kiswahili names where known.
- `species_info/` covers all 547 species; `scripts/species_review_list.py` lists what
  still needs checking, most-photographed species first.
- Size fixes for the Common Ostrich and 9 more species whose length and wingspan were
  mixed up.

## 2.1: in use

- 547 species. Trained on the birds boxed in each photo, with the top MobileNetV2 layers
  fine-tuned.
- Accuracy 60.8%, top-5 80.4% on 7,815 test photos.
- Replaced 2.0 after passing the promotion rules
  ([reports/2.1/promotion_decision.json](reports/2.1/promotion_decision.json)).

## 2.0

- 547 species: every Kenyan species with at least 10 usable photos.
- Whole photos, frozen MobileNetV2 (`--cached-features`).
- Accuracy 44.9%, top-5 69.3%.

## 1.1

- Same 467 species and test photos as 1.0, trained on the boxed birds
  (`data/annotations/`) with fine-tuning.
- Photos with no usable bird, such as sound spectrograms, excluded.
- Accuracy 63.2%, top-5 83.0%.

## 1.0

- 467 species: every Kenyan species with at least 20 usable photos after license checks.
- Whole photos, frozen MobileNetV2.
- Accuracy 44.9%, top-5 69.3%.

## 0.9: baseline

- The 50 Kenyan species with the most GBIF photos.
- Accuracy 72.2%, top-5 90.1%.
