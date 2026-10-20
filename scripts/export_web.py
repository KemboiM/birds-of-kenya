#!/usr/bin/env python3
"""
Build the phone web app's model files (works in Safari on iPhone and Chrome on Android).

    python3 scripts/export_web.py --version 1.0

Writes into app/web/ (git-ignored except index.html and README.md):
    models/classifier.onnx.b64.txt   species classifier (ONNX, base64 text)
    models/detector.onnx.b64.txt     YOLO bird detector with built-in NMS (ONNX, base64 text)
    models/labels.txt                species labels in output order
    species_info.json                common names, sizes etc. from species_info/ (for distance)
    ort/ort-wasm-simd-threaded.*     ONNX Runtime Web engine files (downloaded once)

The models are written as base64 text so that hosts which serve only text and
web files can serve them; any static host (e.g. GitHub Pages) works.
Needs: pip install -r requirements-export.txt
"""

import argparse
import base64
import glob
import json
import os
import shutil
import sys
import urllib.request

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT = "birds"
APP = os.path.join("app", "web")
ORT_VERSION = "1.20.1"          # must match the script tag in app/web/index.html
ORT_FILES = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]
INFO_FIELDS = ["label", "scientific_name", "common_name", "other_names", "size", "can_fly"]


def write_b64(src, dest):
    with open(src, "rb") as f, open(dest, "w") as out:
        out.write(base64.b64encode(f.read()).decode("ascii"))
    print(f"  {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def export_classifier(version, out_dir):
    import keras
    import tensorflow as tf
    import tf2onnx

    src = os.path.join("model", "model_assets", "trained_checkpoints", version,
                       f"{PROJECT}_classify_{version}.keras")
    if not os.path.exists(src):
        sys.exit(f"Classifier not found: {src}")
    trained = keras.models.load_model(src)
    inputs = keras.Input(shape=trained.input_shape[1:])
    x = inputs
    for layer in trained.layers[1:]:
        if layer.name != "augmentation":       # training-only layers
            x = layer(x)
    model = keras.Model(inputs, x)
    size = trained.input_shape[1]
    spec = (tf.TensorSpec((None, size, size, 3), tf.float32, name="image"),)

    @tf.function(input_signature=spec)
    def serve(image):
        return model(image, training=False)

    onnx_path = os.path.join(out_dir, "classifier.onnx")
    tf2onnx.convert.from_function(serve, input_signature=spec, opset=17, output_path=onnx_path)
    write_b64(onnx_path, os.path.join(out_dir, "classifier.onnx.b64.txt"))
    os.remove(onnx_path)
    shutil.copy2(os.path.join("model", "configs_and_labels", f"{PROJECT}_label_list_{version}.txt"),
                 os.path.join(out_dir, "labels.txt"))


def export_detector(weights, out_dir):
    from ultralytics import YOLO
    path = YOLO(weights).export(format="onnx", imgsz=640, nms=True, opset=17, simplify=True)
    write_b64(path, os.path.join(out_dir, "detector.onnx.b64.txt"))


def bundle_species_info():
    info = {}
    for path in glob.glob(os.path.join("species_info", "*.json")):
        with open(path) as f:
            entry = json.load(f)
        info[entry.get("label") or os.path.basename(path)[:-5]] = {k: entry.get(k) for k in INFO_FIELDS}
    with open(os.path.join(APP, "species_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False)
    print(f"  {APP}/species_info.json ({len(info)} species)")


def fetch_runtime():
    ort_dir = os.path.join(APP, "ort")
    os.makedirs(ort_dir, exist_ok=True)
    for name in ORT_FILES:
        dest = os.path.join(ort_dir, name)
        if not os.path.exists(dest):
            url = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/{name}"
            urllib.request.urlretrieve(url, dest)
        print(f"  {dest}")


def main():
    p = argparse.ArgumentParser(description="Build the phone web app's model files.")
    p.add_argument("--version", default="1.0")
    p.add_argument("--detector-weights", default="yolo11n.pt")
    args = p.parse_args()

    out_dir = os.path.join(APP, "models")
    os.makedirs(out_dir, exist_ok=True)
    print("Detector:")
    export_detector(args.detector_weights, out_dir)       # PyTorch before TensorFlow
    print("Classifier:")
    export_classifier(args.version, out_dir)
    print("Species info:")
    bundle_species_info()
    print("Runtime:")
    fetch_runtime()
    print(f"\nDone. Update the version chips in {APP}/index.html, then publish app/web "
          "(see app/web/README.md).")


if __name__ == "__main__":
    main()
