"""Load a folder-per-species image dataset into train/val/test tf.data pipelines.

Expected layout (what scripts/download_gbif_images.py produces):

    data/images/sorted_data/
        Haliaeetus_vocifer/Haliaeetus_vocifer_1.jpg
        Tauraco_hartlaubi/Tauraco_hartlaubi_1.jpg
        ...

Each sub-folder is one class. The split is stratified, so every species
appears in train, validation and test in the same proportions.
"""

import os
from dataclasses import dataclass, field

import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
AUTOTUNE = tf.data.AUTOTUNE


def _is_valid_image(path):
    """Skip truncated or non-image files so training doesn't crash mid-epoch."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def list_images(data_dir, min_images=10, verify=True):
    """Return (paths, labels, class_names) for every class with enough images."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    )

    paths, labels, class_names, skipped = [], [], [], []
    for name in class_dirs:
        folder = os.path.join(data_dir, name)
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if verify:
            files = [f for f in files if _is_valid_image(f)]
        if len(files) < min_images:
            skipped.append((name, len(files)))
            continue
        label = len(class_names)
        class_names.append(name)
        paths.extend(files)
        labels.extend([label] * len(files))

    for name, n in skipped:
        print(f"  Skipping {name}: only {n} usable images (need {min_images})")
    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 species with >= {min_images} images in {data_dir}; "
            f"found {len(class_names)}. Download more with scripts/download_gbif_images.py."
        )
    return paths, labels, class_names


def _load_image(path, label, img_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image.set_shape((*img_size, 3))
    return image, label   # raw 0-255 floats; the model applies MobileNetV2 preprocessing


def _make_dataset(paths, labels, img_size, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load_image(p, l, img_size), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


@dataclass
class Splits:
    """Datasets plus the file lists behind them (test order matches test_ds)."""
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list
    files: dict = field(default_factory=dict)   # {"train"|"val"|"test": (paths, labels)}


def save_split_lists(splits, out_dir, name, version, data_dir):
    """Write train/val/test file lists (like groundnut's trainval files), LF line endings."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for split, (paths, labels) in splits.files.items():
        path = os.path.join(out_dir, f"{split}_{name}_{version}.txt")
        with open(path, "w", newline="\n") as f:
            for p, y in zip(paths, labels):
                f.write(f"{os.path.relpath(p, data_dir)}\t{splits.class_names[y]}\n")
        written.append(path)
    return written


def prepare_dataset(data_dir, img_size=(224, 224), batch_size=32,
                    val_split=0.15, test_split=0.15, min_images=10, seed=42):
    """Build stratified train/val/test datasets.

    Returns a Splits object: batched tf.data.Dataset of (image, label) for each
    split, class_names (species folder names, index = label) and the file lists.
    """
    paths, labels, class_names = list_images(data_dir, min_images=min_images)

    train_p, rest_p, train_y, rest_y = train_test_split(
        paths, labels, test_size=val_split + test_split,
        stratify=labels, random_state=seed)
    val_p, test_p, val_y, test_y = train_test_split(
        rest_p, rest_y, test_size=test_split / (val_split + test_split),
        stratify=rest_y, random_state=seed)

    print(f"  {len(class_names)} species, {len(paths)} images "
          f"(train {len(train_p)}, val {len(val_p)}, test {len(test_p)})")

    train_ds = _make_dataset(train_p, train_y, img_size, batch_size, True, seed)
    val_ds = _make_dataset(val_p, val_y, img_size, batch_size, False, seed)
    test_ds = _make_dataset(test_p, test_y, img_size, batch_size, False, seed)
    return Splits(train_ds, val_ds, test_ds, class_names,
                  {"train": (train_p, train_y), "val": (val_p, val_y), "test": (test_p, test_y)})
