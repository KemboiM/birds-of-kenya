#!/usr/bin/env python3
"""
Export the trained models to TFLite for Android and iOS apps.

    python3 scripts/export_tflite.py --version 1.0              # classifier only
    python3 scripts/export_tflite.py --version 1.0 --detector   # also the YOLO bird detector

Writes to model/model_assets/tflite_models/:
    birds_classify_<version>.tflite      species classifier (float16 weights by default)
    birds_label_list_<version>.txt       labels in output order
    birds_detect_<version>.tflite        bird detector (with --detector)
"""

import argparse
import os
import shutil
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT = "birds"
OUT_DIR = os.path.join("model", "model_assets", "tflite_models")


def export_classifier(version, quantize):
    import keras
    import numpy as np
    import tensorflow as tf

    src = os.path.join("model", "model_assets", "trained_checkpoints", version,
                       f"{PROJECT}_classify_{version}.keras")
    if not os.path.exists(src):
        sys.exit(f"Classifier not found: {src}")
    trained = keras.models.load_model(src)
    # Rebuild without the augmentation block: it only acts during training, and
    # its random ops are not supported by TFLite.
    inputs = keras.Input(shape=trained.input_shape[1:])
    x = inputs
    for layer in trained.layers[1:]:
        if layer.name != "augmentation":
            x = layer(x)
    model = keras.Model(inputs, x)
    saved =os.path.join("model", "model_assets", "trained_checkpoints", version, "saved_model")
    model.export(saved, verbose=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved)
    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{PROJECT}_classify_{version}.tflite")
    with open(out, "wb") as f:
        f.write(tflite)
    labels = os.path.join("model", "configs_and_labels", f"{PROJECT}_label_list_{version}.txt")
    shutil.copy2(labels, OUT_DIR)

    # sanity check: TFLite and Keras agree on a random image
    size = model.input_shape[1]
    x = np.random.uniform(0, 255, (1, size, size, 3)).astype("float32")
    interp = tf.lite.Interpreter(model_path=out)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    diff = np.abs(interp.get_tensor(interp.get_output_details()[0]["index"])
                  - model.predict(x, verbose=0)).max()
    print(f"Classifier: {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"max difference vs Keras {diff:.4f})")


def export_detector(version, weights, imgsz):
    from ultralytics import YOLO
    path = YOLO(weights).export(format="tflite", imgsz=imgsz)   # float32 (fp16 is not offered for TFLite)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{PROJECT}_detect_{version}.tflite")
    shutil.copy2(path, out)
    print(f"Detector: {out} (COCO model: keep only class 14 'bird' in the app)")


def main():
    p = argparse.ArgumentParser(description="Export models to TFLite.")
    p.add_argument("--version", default="1.0")
    p.add_argument("--quantize", choices=["float16", "dynamic", "none"], default="float16")
    p.add_argument("--detector", action="store_true", help="also export the YOLO detector")
    p.add_argument("--detector-weights", default="yolo11n.pt")
    p.add_argument("--detector-imgsz", type=int, default=640)
    args = p.parse_args()
    if args.detector:
        export_detector(args.version, args.detector_weights, args.detector_imgsz)
    export_classifier(args.version, args.quantize)


if __name__ == "__main__":
    main()
