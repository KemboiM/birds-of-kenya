# Species knowledge base

One JSON file per species label (`Haliaeetus_vocifer.json`), read by `identify.py`.

Build or update with:

    python3 scripts/build_species_info.py
    python3 scripts/build_species_info.py --confusion data/validation_data/1.0/1.0_results/most_confused_pairs.csv

Text from Wikipedia is CC BY-SA 4.0; the source link is kept in each file's `sources`.
Files are marked `review.auto_generated: true`. Check sizes against a field guide
(distance estimates depend on them), write the `difference` for each look-alike, then set
`review.verified_by` to your name so later runs don't overwrite your edits.
