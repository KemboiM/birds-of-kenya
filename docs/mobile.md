# Using the models on phones

There are two ways to run Birds of Kenya on a phone. Both use the same trained models.

## 1. Web app (Android and iOS, no install)

`app/web/index.html` runs the YOLO detector and the species classifier inside the
phone's browser with ONNX Runtime Web. It works in Chrome on Android and Safari on
iPhone (iOS 16.4 or later), offline after the first load, and photos never leave the phone.
Build and publish it as described in [app/web/README.md](../app/web/README.md).

## 2. Native apps (TFLite)

```bash
python3 scripts/export_tflite.py --version 1.0 --detector
```

writes to `model/model_assets/tflite_models/`:

| File | Input | Output |
|---|---|---|
| `birds_detect_1.0.tflite` | 1 x 3 x 640 x 640 (channels first), RGB scaled to 0-1, letterboxed | 1 x 84 x 8400: per anchor 4 box values (centre x, centre y, width, height) then 80 COCO class scores; use row 18 (class 14, "bird") and apply non-max suppression in the app |
| `birds_classify_1.0.tflite` | 1 x 224 x 224 x 3, RGB **0-255** (scaling is inside the model) | probabilities, one per line of the label list |
| `birds_label_list_1.0.txt` | | species labels in output order |

The same `.tflite` files run on both platforms:

- **Android**: LiteRT (formerly TensorFlow Lite) with the `com.google.ai.edge.litert`
  dependency, or the TensorFlow Lite Task Library. Put the files in `app/src/main/assets/`.
- **iOS**: LiteRT / `TensorFlowLiteSwift` from CocoaPods or Swift Package Manager.
  Add the files to the Xcode project bundle.

Pipeline in the app, matching `identify.py`:

1. Run the detector on the photo (letterboxed to 640 x 640), keep "bird" boxes with score ≥ 0.25.
2. Crop each box with 10% padding, resize to 224 x 224, run the classifier.
3. Report the top species if its probability is ≥ 0.5, otherwise "unidentified bird".
4. Count boxes per species.

## Licensing in apps

The YOLO11 detector is AGPL-3.0. An app that ships it must make its source available
under the AGPL, or get an Ultralytics enterprise license. The classifier was trained on
CC BY-NC photos, so apps using it must be non-commercial. See the Licenses section of the
main README.
