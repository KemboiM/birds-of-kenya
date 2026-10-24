"""Kedro pipelines for Birds of Kenya: data checks, training, evaluation, promotion and review.

The pipelines call the project's own code in src/ (src.data, src.models, ...), so the
repository root is put on the import path here.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quieter TensorFlow, as in main.py
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "0.1.0"
