# Phone app (Android and iOS)

`index.html` is a single-page web app. It runs the bird detector and the species
classifier inside the phone's browser with ONNX Runtime Web, so it works the same in
Chrome on Android and Safari on iPhone. Photos never leave the phone.

## Build the model files

```bash
pip install -r requirements-export.txt
python3 scripts/export_web.py --version 1.0
```

Then edit the version chips near the top of `index.html` (version, species count,
accuracy) to match `reports/<version>/metrics.json`.

## Publish

Any static host works, as long as `index.html`, `models/`, `ort/` and
`species_info.json` are served from the same folder:

- **Claude artifact** (what the current test link uses): ask Claude to republish `app/web`.
- **GitHub Pages**: copy `app/web` to a `gh-pages` branch, or a `docs/` folder on `main`,
  and enable Pages in the repository settings. The model files are about 27 MB, which is
  fine for Pages.

On the phone, open the link and use **Add to Home Screen** (Safari share menu, or the
Chrome menu) to get an app icon.

## Native apps

For a native Android or iOS app, use the TFLite files from
`python3 scripts/export_tflite.py --version 1.0 --detector`; see `docs/mobile.md`.
