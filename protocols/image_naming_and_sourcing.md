# Image naming and sourcing protocol

## Labels
- Every species label is its scientific name with an underscore: `Haliaeetus_vocifer`.
- The label is the folder name, the line in `birds_label_list_<version>.txt`, and the
  file name in `species_info/`. Common names live only in `species_info/<label>.json`.
- No spaces, no capitals other than the genus, no abbreviations (`Haliaeetus_vocifer`,
  never `fish_eagle`, `HV` or `Haliaeetus Vocifer`).

## Image files
- `data/images/sorted_data/<label>/<label>_<n>.jpg`, numbered from 1 with no gaps
  (`Haliaeetus_vocifer_1.jpg`, `Haliaeetus_vocifer_2.jpg`, ...).
- The downloader assigns numbers; never rename by hand. `data/gbif_metadata.csv` maps
  each file to its GBIF record, license and photographer.
- Field photos you take go in `data/images/images_from_field/` first; move them into
  `sorted_data/<label>/` only after the species is confirmed, continuing the numbering.

## Sources and licenses
- GBIF images: CC0, CC BY or CC BY-NC (`--license commercial` keeps only CC0 and CC BY).
- Credit photographers from `gbif_metadata.csv` whenever images or the model are published.
- One photo per GBIF observation, so near-identical shots of the same bird do not end up
  in both training and test sets.

## Versions
- Every training run gets a new version: `1.0`, `1.1`, `2.0` (`python3 main.py --version 1.1`).
- Everything from a run carries that version: model, label list, train/val/test lists,
  `data/validation_data/<version>/<version>_results/`, and TFLite files.
