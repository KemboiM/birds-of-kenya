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


def _load_crop(path, box, label, img_size):
    """Cut one bird out of a photo. box = [ymin, xmin, ymax, xmax] as fractions of the photo."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)[tf.newaxis]
    crop = tf.image.crop_and_resize(image, box[tf.newaxis], [0], img_size)[0]
    crop.set_shape((*img_size, 3))
    return crop, label


def _make_dataset(paths, labels, img_size, batch_size, shuffle, seed, boxes=None):
    if boxes is None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        load = lambda p, l: _load_image(p, l, img_size)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, tf.constant(boxes, tf.float32), labels))
        load = lambda p, b, l: _load_crop(p, b, l, img_size)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def image_dataset(paths, img_size, boxes=None, flip=False, batch_size=64):
    """Images only (no labels), in file order; used to compute cached features."""
    ds = _make_dataset(paths, [0] * len(paths), img_size, batch_size, False, 0, boxes)
    ds = ds.map(lambda x, l: x)
    if flip:
        ds = ds.map(lambda x: tf.image.flip_left_right(x))
    return ds


@dataclass
class Splits:
    """Datasets plus the file lists behind them (test order matches test_ds)."""
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    class_names: list
    files: dict = field(default_factory=dict)   # {"train"|"val"|"test": (paths, labels)}
    boxes: dict = field(default_factory=dict)   # {"train"|...: [[ymin, xmin, ymax, xmax], ...]} for crops
    img_size: tuple = (224, 224)


def save_split_lists(splits, out_dir, name, version, data_dir):
    """Write train/val/test file lists (one image per line), LF line endings."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for split, (paths, labels) in splits.files.items():
        path = os.path.join(out_dir, f"{split}_{name}_{version}.txt")
        boxes = splits.boxes.get(split)
        with open(path, "w", newline="\n") as f:
            for i, (p, y) in enumerate(zip(paths, labels)):
                extra = "\t" + " ".join(f"{v:.4f}" for v in boxes[i]) if boxes else ""
                f.write(f"{os.path.relpath(p, data_dir)}\t{splits.class_names[y]}{extra}\n")
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


def _read_voc_boxes(xml_path):
    """Boxes from a Pascal VOC file as (xmin, ymin, xmax, ymax, width, height), largest first."""
    import xml.etree.ElementTree as ET
    if not os.path.exists(xml_path):
        return [], None
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        boxes.append(tuple(int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")))
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return boxes, (w, h)


def prepare_crop_dataset(data_root, split_from, img_size=(224, 224), batch_size=32,
                         max_boxes=5, pad=0.10, seed=42, exclude=()):
    """Train on the annotated birds instead of whole photos.

    Uses the photo split of an earlier version (data/training_data/trainvals/
    {train,val,test}_birds_<split_from>.txt) so results are directly comparable,
    and the boxes in data/annotations/<Species>/<photo>.xml.

    train / val: every box in a photo, largest first, up to max_boxes (flocks
                 would otherwise dominate); test: only the largest box, so test
                 results are one prediction per photo.
    A photo without boxes is used whole. Photos listed in `exclude` are skipped.
    """
    labels_path = os.path.join("model", "configs_and_labels", f"birds_label_list_{split_from}.txt")
    with open(labels_path) as f:
        class_names = [l.strip() for l in f if l.strip()]
    index = {c: i for i, c in enumerate(class_names)}
    exclude = set(exclude)

    files, boxes_out, stats = {}, {}, {"boxes": 0, "whole_photo": 0, "excluded": 0}
    for split in ("train", "val", "test"):
        lst = os.path.join(data_root, "training_data", "trainvals", f"{split}_birds_{split_from}.txt")
        paths, labels, boxes = [], [], []
        with open(lst) as f:
            for line in f:
                rel, species = line.rstrip("\n").split("\t")[:2]
                if rel in exclude:
                    stats["excluded"] += 1
                    continue
                photo = os.path.join(data_root, rel)
                stem = os.path.splitext(os.path.basename(rel))[0]
                found, size = _read_voc_boxes(os.path.join(data_root, "annotations", species, stem + ".xml"))
                keep = found[:1] if split == "test" else found[:max_boxes]
                if not keep:
                    paths.append(photo); labels.append(index[species]); boxes.append([0.0, 0.0, 1.0, 1.0])
                    stats["whole_photo"] += 1
                    continue
                w, h = size
                for x1, y1, x2, y2 in keep:
                    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
                    boxes.append([max(0.0, (y1 - ph) / h), max(0.0, (x1 - pw) / w),
                                  min(1.0, (y2 + ph) / h), min(1.0, (x2 + pw) / w)])
                    paths.append(photo); labels.append(index[species])
                    stats["boxes"] += 1
        files[split], boxes_out[split] = (paths, labels), boxes

    print(f"  {len(class_names)} species; crops: train {len(files['train'][0])}, "
          f"val {len(files['val'][0])}, test {len(files['test'][0])} "
          f"({stats['boxes']} bird boxes, {stats['whole_photo']} photos used whole, "
          f"{stats['excluded']} photos excluded)")
    ds = {s: _make_dataset(files[s][0], files[s][1], img_size, batch_size, s == "train", seed, boxes_out[s])
          for s in files}
    return Splits(ds["train"], ds["val"], ds["test"], class_names, files, boxes_out, img_size)
