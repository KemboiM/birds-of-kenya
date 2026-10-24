"""Tests for the reliability components. Run from the repository root: pytest tests"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bok_pipelines.reliability import data_checks as dc  # noqa: E402
from bok_pipelines.reliability.evaluation import evaluation_report, expected_calibration_error  # noqa: E402
from bok_pipelines.reliability.gate import promotion_decision  # noqa: E402
from bok_pipelines.reliability.monitoring import drift_report, psi, review_queue  # noqa: E402
from bok_pipelines.reliability.schemas import METADATA, PER_LABEL, schema_issues  # noqa: E402


def make_image(path, color, size=(120, 90), orientation=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.new("RGB", size, color)
    if orientation:
        exif = im.getexif()
        exif[0x0112] = orientation
        im.save(path, exif=exif)
    else:
        im.save(path)


def voc(path, label, size, box):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (w, h), (x1, y1, x2, y2) = size, box
    with open(path, "w") as f:
        f.write(f"<annotation><size><width>{w}</width><height>{h}</height></size><object><name>{label}</name>"
                f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>"
                f"</object></annotation>")


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    make_image(images / "A" / "a1.jpg", (200, 0, 0))
    make_image(images / "A" / "a2.jpg", (190, 10, 0))
    make_image(images / "B" / "b1.jpg", (0, 0, 200))
    make_image(images / "B" / "b2.jpg", (200, 0, 0))             # same pixels as A/a1: conflicting labels
    make_image(images / "B" / "b3.jpg", (0, 10, 190), size=(40, 30))
    with open(images / "B" / "broken.jpg", "wb") as f:
        f.write(b"not an image")
    index = dc.index_images(dc.list_labelled_images(str(images)), root=str(tmp_path), workers=2)
    return tmp_path, index


def test_index_and_basic_checks(dataset):
    _, index = dataset
    assert len(index) == 6
    assert dc.check_readable(index).issues == 1
    assert dc.check_min_size(index, 64, 64).issues == 1
    same, cross = dc.check_duplicates(index)
    assert cross.issues == 2 and not cross.passed
    assert dc.check_label_counts(index, 3).issues == 1        # A has 2 readable photos


def test_licenses(dataset):
    _, index = dataset
    meta = pd.DataFrame({"file": index["file"], "license": ["CC-BY"] * 4 + ["ARR", "CC0"]})
    meta = meta[meta["file"] != index["file"].iloc[0]]          # one photo with no record
    result = dc.check_licenses(index, meta, ["CC0", "CC-BY"])
    assert result.issues == 2
    assert set(result.rows["detail"].str.split().str[0]) == {"not", "license"}


def test_annotations(tmp_path):
    images, ann = tmp_path / "images", tmp_path / "annotations"
    make_image(images / "A" / "ok.jpg", (10, 10, 10), size=(100, 80))
    make_image(images / "A" / "outside.jpg", (10, 10, 10), size=(100, 80))
    make_image(images / "A" / "turned.jpg", (10, 10, 10), size=(100, 80), orientation=6)
    make_image(images / "A" / "nobox.jpg", (10, 10, 10), size=(100, 80))
    voc(ann / "A" / "ok.xml", "A", (100, 80), (10, 10, 50, 50))
    voc(ann / "A" / "outside.xml", "B", (100, 80), (10, 10, 150, 50))   # wrong label and outside
    voc(ann / "A" / "turned.xml", "A", (80, 100), (10, 10, 50, 50))
    index = dc.index_images(dc.list_labelled_images(str(images)), root=str(tmp_path))
    result = dc.check_annotations(index, str(ann))
    details = " | ".join(result.rows["detail"])
    assert result.issues == 3
    assert "box labelled B" in details and "outside the photo" in details and "EXIF orientation 6" in details
    assert "1 without" in result.note


def test_splits(dataset):
    _, index = dataset
    files = dict(zip(index["file"], index["label"]))
    a1, a2 = "images/A/a1.jpg", "images/A/a2.jpg"
    b1, b2 = "images/B/b1.jpg", "images/B/b2.jpg"
    good = {"train": [(a1, "A"), (b1, "B")], "test": [(a2, "A"), (b1.replace("b1", "b3"), "B")]}
    assert dc.check_splits(index, good).passed
    leak = {"train": [(a1, "A"), (b1, "B")], "test": [(a2, "A"), (b2, "B")]}     # b2 has a1's pixels
    assert "identical photo also in train" in " ".join(dc.check_splits(index, leak).rows["detail"])
    twice = {"train": [(a1, "A"), (b1, "B")], "test": [(a1, "A"), (b1, "B")]}
    assert dc.check_splits(index, twice).issues >= 2
    assert dc.check_splits(index, {}).skipped
    assert set(files.values()) == {"A", "B"}


def test_summary_blocking():
    results = [dc.result("licenses", [("f", "A", "bad")]), dc.result("too_small_images", [("g", "A", "small")])]
    summary, issues = dc.summarize(results, {"licenses"})
    assert not summary["passed"] and summary["blocking_failures"] == ["licenses"]
    assert summary["warnings"] == ["too_small_images"] and len(issues) == 2


def test_evaluation_report():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    summary, per_label = evaluation_report([0, 0, 1, 1], probs, ["A", "B"], {"A": 50, "B": 5}, rare_max_train=10,
                                           top_k=5)
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["top2_accuracy"] == 1.0
    assert summary["accuracy_rare_labels"] == pytest.approx(0.5)
    assert summary["accuracy_common_labels"] == pytest.approx(1.0)
    PER_LABEL.validate(per_label)
    perfect = expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert perfect == 0.0


def report(acc, f1, rare=None):
    return {"accuracy": acc, "f1_macro": f1, "accuracy_rare_labels": rare}


def per_label(recalls, n=20):
    return pd.DataFrame({"label": list(recalls), "test_images": n, "recall": list(recalls.values())})


def test_gate():
    ok = {"passed": True, "blocking_failures": []}
    better = promotion_decision(report(0.7, 0.6, 0.4), per_label({"A": 0.8, "B": 0.7}),
                                report(0.6, 0.5, 0.3), per_label({"A": 0.7, "B": 0.7}), ok)
    assert better["promote"]
    worse = promotion_decision(report(0.55, 0.6), None, report(0.6, 0.5), None, ok)
    assert not worse["promote"] and not worse["checks"]["metric_gain"]["passed"]
    regress = promotion_decision(report(0.7, 0.6), per_label({"A": 0.4, "B": 0.9}),
                                 report(0.6, 0.5), per_label({"A": 0.7, "B": 0.7}), ok)
    assert not regress["promote"] and regress["checks"]["label_regressions"]["labels"][0]["label"] == "A"
    bad_data = promotion_decision(report(0.7, 0.6), None, report(0.6, 0.5), None,
                                  {"passed": False, "blocking_failures": ["licenses"]})
    assert not bad_data["promote"]
    no_checks = promotion_decision(report(0.7, 0.6), None, None, None, None, require_data_checks=True)
    assert not no_checks["promote"]


def test_psi_and_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.1
    assert psi(ref, rng.normal(2, 1, 2000)) > 0.25
    assert psi(np.full(100, 0.2), np.full(100, 0.8)) > 0.25      # constant reference still detects a shift
    report_ = drift_report(pd.DataFrame({"x": ref}), pd.DataFrame({"x": ref[:10]}), ["x"])
    assert report_["note"]                                         # too few new images is flagged


def test_review_queue():
    preds = pd.DataFrame({"photo": ["p1", "p1", "p2"], "bird_id": [1, 2, 1], "confidence": [0.9, 0.3, 0.6],
                          "identified": [True, False, False]})
    queue = review_queue(preds, min_confidence=0.5)
    assert list(queue["confidence"]) == [0.3, 0.6]
    assert {"correct_label", "reviewer", "reviewed_on"} <= set(queue.columns)


def test_metadata_schema_issues():
    meta = pd.DataFrame({"file": ["a.jpg", "a.jpg"], "species": ["A", "A"], "license": ["CC0", None]})
    _, rows = schema_issues(METADATA, meta, "metadata")
    assert len(rows) >= 2


def test_pipelines_register():
    pytest.importorskip("kedro")
    from bok_pipelines.pipeline_registry import register_pipelines
    pipelines = register_pipelines()
    assert {"data_checks", "training", "evaluation", "promotion", "review", "release", "__default__"} <= set(pipelines)
    order = [n.name for n in pipelines["release"].nodes]
    assert order.index("run_data_checks") < order.index("train_with_main_py") < order.index("evaluate_version")
    assert order.index("evaluate_version") < order.index("decide_promotion")
