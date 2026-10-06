"""Verifier-side C1 metrics, independent of the editable Python interpreter.

The candidate submits image predictions, NEVER labels, category membership or
AUROC/AP/F1. Counts/order below are the existing frozen category_paths protocol
for archive cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d.
No candidate module, pickle, or candidate-selected file is loaded here.

This closes self-reported metric forgery. It does not claim to solve the
separate, pre-existing disclosure of labels in MVTec filenames/test ordering.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .prediction_frame import read_prediction_frame


# (category, normal images, anomalous images), in frozen manifest order.
CATEGORIES = (
    ("bottle", 20, 63), ("cable", 58, 92), ("capsule", 23, 109),
    ("carpet", 28, 89), ("grid", 21, 57), ("hazelnut", 40, 70),
    ("leather", 32, 92), ("metal_nut", 22, 93), ("pill", 26, 141),
    ("screw", 41, 119), ("tile", 33, 84), ("toothbrush", 12, 30),
    ("transistor", 60, 40), ("wood", 19, 60), ("zipper", 32, 119),
)
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
AXES = {"coreset", "distance-metric", "feature-layer", "score-aggregation",
        "threshold-calibration"}


def trusted_profile(parser_file: str) -> str:
    """Only a protected task/render-copy config may opt into reduced counts.

Never consult worker output, worker env, or an agent-writable result file to
select a smaller evaluation set. The default delivery contract is full.
"""
    config = json.loads(Path(parser_file).with_name("config.json").read_text())
    profile = config.get("anomalib_prediction_profile", "full")
    if profile not in ("full", "smoke"):
        raise ValueError("unknown protected anomalib_prediction_profile")
    return profile


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is out of range") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _ranking(score, truth):
    # Deliberately preserve the old stable tie ordering and trapezoid formula;
    # changing metric definitions would invalidate the existing anchors.
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives or not negatives:
        raise ValueError("ranking needs both classes")
    ranked = truth[np.argsort(-score, kind="mergesort")]
    tp, fp = np.cumsum(ranked), np.cumsum(~ranked)
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    precision = tp / np.arange(1, len(tp) + 1)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return {"image_auroc": float(integrate(tpr, fpr)),
            "image_ap": float(precision[ranked].sum() / positives)}


def _threshold_metrics(score, truth, threshold):
    prediction = score >= threshold
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"decision_f1": 2 * precision * recall / max(1e-12, precision + recall),
            "balanced_accuracy": 0.5 * (recall + tn / max(1, tn + fp)),
            "normal_fpr": fp / max(1, fp + tn), "anomaly_recall": recall}


def score_predictions(raw_output: str, *, axis: str, profile: str) -> dict:
    if axis not in AXES or profile not in ("full", "smoke"):
        raise ValueError("unknown trusted prediction contract")
    frame = read_prediction_frame(raw_output)
    expected = {"scores", "runtime_seconds"}
    if axis == "threshold-calibration":
        expected.add("threshold")
    if set(frame) != expected:
        raise ValueError("unexpected prediction fields (labels/metrics/profile are not accepted)")
    groups = frame["scores"]
    if not isinstance(groups, list) or len(groups) != len(CATEGORIES):
        raise ValueError("expected predictions for exactly 15 categories")
    scores, labels = [], []
    for group, (category, good, bad) in zip(groups, CATEGORIES):
        if profile == "smoke":
            good, bad = min(good, 4), min(bad, 4)
        if axis == "threshold-calibration":
            good -= max(1, good // 2)
            bad -= max(1, bad // 2)
        if not isinstance(group, list) or len(group) != good + bad:
            raise ValueError(f"wrong prediction count for {category}; expected {good + bad}")
        scores.append(np.asarray([_scalar(v, "image prediction") for v in group],
                                 dtype=np.float64))
        labels.append(np.asarray([False] * good + [True] * bad))

    texture = [i for i, (name, _, _) in enumerate(CATEGORIES) if name in TEXTURES]
    objects = [i for i, (name, _, _) in enumerate(CATEGORIES) if name not in TEXTURES]
    partitions = {"texture": texture, "object_a": objects[:5], "object_b": objects[5:]}
    if axis in ("coreset", "distance-metric", "feature-layer"):
        rows = [_ranking(s, y) for s, y in zip(scores, labels)]
        metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for name, indices in partitions.items():
            metrics.update({f"{name}_{key}": float(np.mean([rows[i][key] for i in indices]))
                            for key in rows[0]})
    else:
        threshold = _scalar(frame["threshold"], "threshold") if "threshold" in frame else None
        def evaluate(s, y):
            return _ranking(s, y) if threshold is None else _threshold_metrics(s, y, threshold)
        metrics = evaluate(np.concatenate(scores), np.concatenate(labels))
        for name, indices in partitions.items():
            row = evaluate(np.concatenate([scores[i] for i in indices]),
                           np.concatenate([labels[i] for i in indices]))
            metrics.update({f"{name}_{key}": value for key, value in row.items()})
    metrics = {key: float(f"{value:.8f}") for key, value in metrics.items()}
    # Legacy diagnostic only: NO C1 score_spec consumes this self-reported time.
    elapsed = _scalar(frame["runtime_seconds"], "runtime_seconds")
    if elapsed < 0:
        raise ValueError("runtime_seconds must be nonnegative")
    metrics["runtime_seconds"] = float(f"{elapsed:.6f}")
    return metrics
