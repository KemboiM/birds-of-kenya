"""Training: runs main.py with the settings in parameters.yml, after the data checks pass."""

import logging
import os
import subprocess
import sys

from ... import REPO_ROOT
from ...project import as_version, model_path, read_json

log = logging.getLogger(__name__)


def main_py_args(version, t):
    args = ["--version", version, "--epochs", str(t["epochs"]), "--fine-tune-epochs", str(t["fine_tune_epochs"]),
            "--batch-size", str(t["batch_size"]), "--img-size", str(t["img_size"]),
            "--min-images", str(t["min_images"])]
    if t.get("from_annotations"):
        args += ["--from-annotations", "--split-from", as_version(t["split_from"])]
    if t.get("exclude") and os.path.exists(t["exclude"]):
        args += ["--exclude", t["exclude"]]
    if t.get("cached_features"):
        args.append("--cached-features")
    return args + [str(a) for a in t.get("extra_args", [])]


def train(data_checks, version, paths, t):
    version = as_version(version)
    if t.get("require_data_checks", True) and not data_checks.get("passed"):
        raise RuntimeError(f"Blocking data checks failed for {version}: "
                           f"{', '.join(data_checks.get('blocking_failures', []))}. Fix them, or set "
                           f"training.require_data_checks: false in conf/base/parameters.yml.")
    if t.get("require_data_checks", True) and data_checks.get("subset"):
        raise RuntimeError(f"The data checks for {version} ran on a trial subset. Run them on every photo first: "
                           f"kedro run --pipelines data_checks --params version={version}")
    target = model_path(paths, version)
    if os.path.exists(target):
        raise RuntimeError(f"Version {version} already exists ({target}). Use a new version, e.g. "
                           f"kedro run --pipeline release --params version=<new>")
    cmd = [sys.executable, "main.py", *main_py_args(version, t)]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not os.path.exists(target):
        raise RuntimeError(f"main.py finished but {target} was not written; see its output above.")
    return read_json(os.path.join(paths["labels"], f"train_config_{version}.json"))
