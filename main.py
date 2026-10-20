import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data.loader import prepare_dataset
from src.models.train import MODEL_PATH, evaluate_model, train_model
from src.utils.helpers import log_training, save_class_names, save_metrics, save_report


def parse_args():
    p = argparse.ArgumentParser(description="Train a Kenyan bird species classifier.")
    p.add_argument('--data-dir', default='data/gbif/images',
                   help='folder with one sub-folder of images per species')
    p.add_argument('--epochs', type=int, default=10, help='head-training epochs')
    p.add_argument('--fine-tune-epochs', type=int, default=10,
                   help='fine-tuning epochs after the head is trained (0 to skip)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--min-images', type=int, default=10,
                   help='skip species with fewer usable images than this')
    p.add_argument('--no-pretrained', action='store_true',
                   help='random init instead of ImageNet weights (offline testing only)')
    return p.parse_args()


def main():
    """Main pipeline for training the multi-species bird classifier."""
    args = parse_args()
    run_name = 'kenya-birds'

    print("\n" + "=" * 60)
    print("Bird Classifier: KENYAN BIRD SPECIES")
    print("=" * 60)

    print(f"\nLooking for images in: {args.data_dir}")
    if not os.path.exists(args.data_dir):
        print("ERROR: Directory not found!")
        print("Download images first with:")
        print("  python3 scripts/download_gbif_images.py --top-species 50 --max-per-species 200")
        return

    print("\nStep 1: Loading images...")
    train_ds, val_ds, test_ds, class_names = prepare_dataset(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        min_images=args.min_images,
    )
    save_class_names(class_names)

    print("\nStep 2: Training model...")
    model, history = train_model(
        train_ds, val_ds, num_classes=len(class_names),
        epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs,
        weights=None if args.no_pretrained else 'imagenet',
        img_size=args.img_size,
    )

    print("\nStep 3: Evaluating model...")
    metrics, report = evaluate_model(model, test_ds, class_names)

    print("\nStep 4: Saving results...")
    save_metrics(metrics, run_name)
    save_report(report, run_name)
    log_training(run_name, args.epochs + args.fine_tune_epochs, args.batch_size, metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print("Label order saved to: models/class_names.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
