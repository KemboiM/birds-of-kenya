"""Train the Kenyan bird species classifier (stage 2 of the identification tool).

Inputs : data/images/sorted_data/<Scientific_name>/<Scientific_name>_N.jpg
Outputs (for --version 1.0):
    model/model_assets/trained_checkpoints/1.0/birds_classify_1.0.keras
    model/configs_and_labels/birds_label_list_1.0.txt    (+ birds_label_list.txt = latest)
    model/configs_and_labels/train_config_1.0.json
    splits/{train,val,test}_birds_1.0.txt
    data/validation_data/1.0/1.0_results/                (N x N confusion matrix, misclassified/, ...)
    reports/1.0/                                         small copies of the results, committed to git
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime

# Results small enough to commit (the image folders stay in data/validation_data)
REPORT_FILES = ['metrics.json', 'per_species_2x2.csv', 'most_confused_pairs.csv',
                'confusion_matrix.csv', 'confusion_matrix.png', 'training_curves.png']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data.loader import prepare_crop_dataset, prepare_dataset, save_split_lists
from src.evaluation.report import write_evaluation
from src.models.train import predict_dataset, train_model, train_model_cached
from src.utils.helpers import log_training, save_label_list

PROJECT = 'birds'


def parse_args():
    p = argparse.ArgumentParser(description="Train the Kenyan bird species classifier.")
    p.add_argument('--version', default='1.0', help='model version, e.g. 1.0, 1.1, 2.0')
    p.add_argument('--data-root', default='data')
    p.add_argument('--data-dir', default=None,
                   help='species folders (default: <data-root>/images/sorted_data)')
    p.add_argument('--epochs', type=int, default=10, help='head-training epochs')
    p.add_argument('--fine-tune-epochs', type=int, default=10,
                   help='fine-tuning epochs after the head is trained (0 to skip)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--min-images', type=int, default=10,
                   help='skip species with fewer usable images than this')
    p.add_argument('--cached-features', action='store_true',
                   help='fast CPU training: compute MobileNetV2 features once (photo + mirrored copy) '
                        'and train only the classifier head; no fine-tuning')
    p.add_argument('--from-annotations', action='store_true',
                   help='train on the boxed birds in data/annotations instead of whole photos')
    p.add_argument('--split-from', default='1.0',
                   help='with --from-annotations: reuse this version\'s train/val/test photo split')
    p.add_argument('--exclude', default=None,
                   help='CSV with an "image" column of photos to leave out (e.g. non-bird images)')
    p.add_argument('--copy-correct', action='store_true',
                   help='also copy correctly classified test images into correct/')
    p.add_argument('--no-pretrained', action='store_true',
                   help='random init instead of ImageNet weights (offline testing only)')
    return p.parse_args()


def main():
    args = parse_args()
    v = args.version
    data_dir = args.data_dir or os.path.join(args.data_root, 'images', 'sorted_data')
    checkpoint_dir = os.path.join('model', 'model_assets', 'trained_checkpoints', v)
    model_path = os.path.join(checkpoint_dir, f'{PROJECT}_classify_{v}.keras')
    config_dir = os.path.join('model', 'configs_and_labels')
    results_dir = os.path.join(args.data_root, 'validation_data', v, f'{v}_results')

    print("\n" + "=" * 60)
    print(f"Bird Classifier: KENYAN BIRD SPECIES  (version {v})")
    print("=" * 60)

    print(f"\nLooking for images in: {data_dir}")
    if not os.path.exists(data_dir):
        print("ERROR: Directory not found!")
        print("Download images first with:  python3 scripts/download_gbif_images.py")
        return
    if os.path.exists(model_path):
        print(f"ERROR: version {v} already exists ({model_path}).")
        print("Use a new --version, e.g. --version 1.1, so earlier results are kept.")
        return

    print("\nStep 1: Loading images...")
    exclude = []
    if args.exclude:
        with open(args.exclude, newline='') as f:
            exclude = [row['image'] for row in csv.DictReader(f)]
    if args.from_annotations:
        splits = prepare_crop_dataset(args.data_root, args.split_from,
                                      img_size=(args.img_size, args.img_size),
                                      batch_size=args.batch_size, exclude=exclude)
    else:
        if exclude:
            print(f"  Leaving out {len(exclude)} photos listed in {args.exclude}")
        splits = prepare_dataset(data_dir, img_size=(args.img_size, args.img_size),
                                 batch_size=args.batch_size, min_images=args.min_images,
                                 exclude=[os.path.join(args.data_root, e) for e in exclude])
    names = splits.class_names
    save_label_list(names, config_dir, PROJECT, v)
    for path in save_split_lists(splits, 'splits',
                                 PROJECT, v, args.data_root):
        print(f"  Split list saved to {path}")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, f'train_config_{v}.json'), 'w') as f:
        json.dump({**vars(args), 'num_classes': len(names), 'model_path': model_path,
                   'base_model': 'MobileNetV2', 'created': datetime.now().isoformat()}, f, indent=2)

    print("\nStep 2: Training model...")
    weights = None if args.no_pretrained else 'imagenet'
    if args.cached_features:
        model, history = train_model_cached(
            splits, num_classes=len(names), epochs=args.epochs, weights=weights,
            img_size=args.img_size, model_path=model_path,
            cache_path=os.path.join(args.data_root, 'training_data', f'feature_cache_{v}.npz'))
    else:
        model, history = train_model(
            splits.train_ds, splits.val_ds, num_classes=len(names),
            epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs,
            weights=weights, img_size=args.img_size, model_path=model_path,
        )

    print("\nStep 3: Evaluating on the test images...")
    y_true, probs = predict_dataset(model, splits.test_ds)
    metrics = write_evaluation(results_dir, y_true, probs, names,
                               test_paths=splits.files['test'][0], history=history,
                               data_root=args.data_root, copy_correct=args.copy_correct)

    print("\nStep 4: Saving results...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy2(os.path.join(config_dir, f'{PROJECT}_label_list_{v}.txt'), checkpoint_dir)
    epochs_run = len(history.history.get('loss', []))
    log_training(f'{PROJECT}_{v}', epochs_run, args.batch_size, metrics)
    report_dir = os.path.join('reports', v)
    os.makedirs(report_dir, exist_ok=True)
    for name in REPORT_FILES:
        if os.path.exists(os.path.join(results_dir, name)):
            shutil.copy2(os.path.join(results_dir, name), report_dir)
    print(f"Report for git: {report_dir}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model:       {model_path}")
    print(f"Labels:      {config_dir}/{PROJECT}_label_list_{v}.txt")
    print(f"Evaluation:  {results_dir}")
    print(f"Next: python3 scripts/export_tflite.py --version {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
