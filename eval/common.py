"""Shared helpers for the SentinelAI evaluation pipeline.

Everything that must be identical across experiments lives here: paths, the
global seed, metric computation, threshold selection (VAL only) and latency
statistics.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
ARTIFACTS = REPORTS / "artifacts"  # per-email scores (git-ignored)

SEED = 42
TORCH_THREADS = 4  # fixed: float reductions can differ with thread count

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # Windows consoles/pipes default to cp1252
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(TORCH_THREADS)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def ranking_metrics(y_true, scores) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }


def full_metrics(y_true, scores, threshold: float) -> dict:
    """Binary metrics at `score >= threshold` plus threshold-free ranking metrics."""
    scores = np.asarray(scores, dtype=float)
    out = binary_metrics(y_true, scores >= threshold)
    out.update(ranking_metrics(y_true, scores))
    out["threshold"] = float(threshold)
    return out


def select_threshold_on_val(y_val, val_scores) -> float:
    """Return the threshold maximising F1 on the VALIDATION split.

    Candidates are the distinct observed scores; ties broken towards the higher
    threshold (fewer false positives). Never call this with test data.
    """
    y_val = np.asarray(y_val).astype(int)
    val_scores = np.asarray(val_scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)[:-1]
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)[-1]
    return float(thresholds[best])


def latency_stats(seconds: list[float]) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1000.0
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "n_timed": int(len(ms)),
    }


def dir_size_mb(path: Path, pattern: str = "*") -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob(pattern) if p.is_file()) / 1e6


def latency_sample_ids(test_df, n: int = 1000) -> list[int]:
    """The same seeded sample of test emails is timed for every approach."""
    n = min(n, len(test_df))
    return test_df.sample(n=n, random_state=SEED)["row_id"].tolist()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
