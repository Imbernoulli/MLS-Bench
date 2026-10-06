"""Recompute CV metrics in the protected parser, never in candidate Python.

The wire format contains predictions only. Targets, settings and formulae come
from the protected task/runner tree. This prevents self-reported metric forgery;
it is not a sandbox or a claim that the public datasets have secret labels.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .prediction_frame import read_prediction_frame

TARGET_SHA256 = "829f9ab154fa32d346eaa47dc43c7d5a96f0f8fd1b805bcffd58b1bb91782e52"
GROUPS = ("congeneric_large", "congeneric_small", "distinctive")


@lru_cache(maxsize=1)
def frozen_targets():
    raw = (Path(__file__).parent / "assets/cv_prediction_targets_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != TARGET_SHA256:
        raise ValueError("protected CV target metadata SHA256 mismatch")
    return json.loads(raw)


def _balanced_labels(labels, limit):
    if limit == len(labels):
        return labels
    classes = sorted(set(labels))[:min(len(set(labels)), max(2, limit // 4))]
    counts = {label: labels.count(label) for label in classes}
    result, depth = [], 0
    while len(result) < limit:
        added = False
        for label in classes:
            if depth < counts[label] and len(result) < limit:
                result.append(label)
                added = True
        if not added:
            raise ValueError("protected retrieval sample limit cannot be balanced")
        depth += 1
    return result


def protected_labels(family, parser_file):
    key = "pml" if family.startswith("pml_") else family
    if key not in ("crowd", "pointnext", "pml"):
        raise ValueError("unknown prediction family")
    labels = frozen_targets()[key]["targets"]
    config = json.loads(Path(parser_file).with_name("config.json").read_text())
    protocol = config.get("prediction_evaluation", {"profile": "formal"})
    if not isinstance(protocol, dict) or set(protocol) - {"profile", "sample_limit"}:
        raise ValueError("invalid protected prediction evaluation configuration")
    profile = protocol.get("profile", "formal")
    limit = protocol.get("sample_limit", len(labels))
    if type(limit) is not int or not 1 <= limit <= len(labels):
        raise ValueError("invalid protected evaluation sample limit")
    if profile not in ("formal", "smoke"):
        raise ValueError("unknown protected evaluation profile")
    if profile == "formal" and limit != len(labels):
        raise ValueError("formal scoring requires the full frozen evaluation set")
    selected = _balanced_labels(labels, limit) if key == "pml" else labels[:limit]
    return np.asarray(selected, dtype=np.int64), profile == "formal"


def _vector(value, size, *, integer=False):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"expected {size} predictions")
    if any(type(x) not in ((int,) if integer else (int, float)) for x in value):
        raise ValueError("predictions have an invalid scalar type")
    try:
        result = np.asarray(value, dtype=np.int64 if integer else np.float64)
    except (OverflowError, ValueError) as exc:
        raise ValueError("prediction outside numeric range") from exc
    if not np.isfinite(result).all():
        raise ValueError("non-finite prediction")
    return result


def _crowd(frame, gold, formal):
    if set(frame) - {"predictions", "diagnostics"}:
        raise ValueError("unknown crowd prediction field")
    errors = _vector(frame.get("predictions"), len(gold)) - gold
    metrics = {"count_mae": float(np.mean(np.abs(errors))),
               "count_rmse": float(np.sqrt(np.mean(np.square(errors)))),
               "evaluation_images": float(len(gold))}
    groups = np.where(gold < 232, 0, np.where(gold < 427, 1, 2))
    for index, name in enumerate(("sparse", "medium", "dense")):
        values = errors[groups == index]
        if formal and len(values) < 10:
            raise ValueError("incomplete formal crowd density group")
        if not len(values):
            continue
        metrics[name + "_mae"] = float(np.mean(np.abs(values)))
        metrics[name + "_rmse"] = float(np.sqrt(np.mean(np.square(values))))
    # These are untrusted diagnostics, not inputs to any of the crowd score specs.
    diagnostics = frame.get("diagnostics", {})
    if not isinstance(diagnostics, dict) or set(diagnostics) - {
        "train_updates", "train_objective", "runtime_seconds"
    }:
        raise ValueError("unknown crowd diagnostic")
    for name, value in diagnostics.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid crowd diagnostic")
        metrics[name] = float(value)
    return metrics


def _pointnext(frame, gold, formal):
    if set(frame) != {"predictions"}:
        raise ValueError("unknown point classification prediction field")
    prediction = _vector(frame["predictions"], len(gold), integer=True)
    if np.any(prediction < 0) or np.any(prediction >= 15):
        raise ValueError("point class outside [0, 15)")
    confusion = np.bincount(gold * 15 + prediction, minlength=225).reshape(15, 15)
    correct, total = confusion.diagonal(), confusion.sum(axis=1)
    valid = total > 0
    metrics = {"oa": float(correct.sum() / total.sum()),
               "macc": float((correct[valid] / total[valid]).mean())}
    for index, name in enumerate(("class_a", "class_b", "class_c")):
        block = slice(index * 5, (index + 1) * 5)
        n, c = total[block], correct[block]
        if n.sum() == 0:
            if formal:
                raise ValueError("incomplete formal point classification group")
            continue
        mask = n > 0
        metrics[name + "_oa"] = float(c.sum() / n.sum())
        metrics[name + "_macc"] = float((c[mask] / n[mask]).mean())
    return metrics


def decode_rankings(encoded, gold):
    if not isinstance(encoded, str):
        raise ValueError("rankings must be a base64 string")
    counts = np.bincount(gold)[gold] - 1
    width = int(counts.max())
    if width < 1:
        raise ValueError("retrieval requires repeated classes")
    expected_bytes = len(gold) * width * 2
    if len(encoded) != 4 * ((expected_bytes + 2) // 3):
        raise ValueError("ranking payload length does not match protected evaluation set")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid ranking base64") from exc
    if len(raw) != expected_bytes:
        raise ValueError("truncated ranking payload")
    ranking = np.frombuffer(raw, dtype="<u2").reshape(len(gold), width).astype(np.int64)
    if (ranking >= len(gold)).any() or (ranking == np.arange(len(gold))[:, None]).any():
        raise ValueError("out-of-range or self retrieval neighbor")
    if (np.diff(np.sort(ranking, axis=1), axis=1) == 0).any():
        raise ValueError("duplicate retrieval neighbor")
    return ranking


def retrieval_metrics(ranking, gold, groups=None, *, formal=True):
    # Preserve the original evaluator's float32 reductions and class averaging.
    # Explicit CPU tensors: the private parser does not allocate a GPU.
    import torch

    labels = torch.tensor(gold, dtype=torch.long, device="cpu")
    ranks = torch.tensor(ranking, dtype=torch.long, device="cpu")
    counts = torch.bincount(labels)[labels] - 1
    width = ranks.shape[1]
    matches = labels[ranks].eq(labels[:, None])
    r1 = matches[:, :1].any(1).float()
    r2 = matches[:, :min(2, width)].any(1).float()
    r4 = matches[:, :min(4, width)].any(1).float()
    positions = torch.arange(1, width + 1, dtype=torch.float32, device="cpu")[None]
    precision = matches.cumsum(1) / positions
    ap = (precision * matches * (positions <= counts[:, None])).sum(1) / counts.clamp_min(1)

    def summarise(mask):
        y, a, b, c, d = labels[mask], r1[mask], r2[mask], r4[mask], ap[mask]
        class_means = torch.stack([a[y == label].mean() for label in torch.unique(y)])
        decile = max(1, math.ceil(len(class_means) * 0.10))
        return {"recall_at_1": float(a.mean()), "recall_at_2": float(b.mean()),
                "recall_at_4": float(c.mean()), "map_at_r": float(d.mean()),
                "worst_decile_recall_at_1": float(class_means.sort().values[:decile].mean())}

    metrics = summarise(torch.ones(len(labels), dtype=torch.bool, device="cpu"))
    if groups is not None:
        group_tensor = torch.tensor(groups, dtype=torch.long, device="cpu")
        for index, name in enumerate(GROUPS):
            mask = group_tensor == index
            if int(mask.sum()) < 2:
                if formal:
                    raise ValueError("incomplete formal retrieval taxonomy group")
                continue
            metrics.update({name + "_" + k: v for k, v in summarise(mask).items()})
    return metrics


def _pml(frame, gold, family, formal):
    if set(frame) != {"rankings"} or not isinstance(frame["rankings"], dict):
        raise ValueError("expected only retrieval rankings")
    expected = {"epochs10", "epochs20", "epochs40"} if family == "pml_budget" else {"final"}
    if set(frame["rankings"]) != expected:
        raise ValueError("missing or extra protected retrieval setting")
    groups = None
    if family == "pml_taxonomy":
        groups = np.asarray(frozen_targets()["pml"]["groups"], dtype=np.int64)[gold]
    metrics = {}
    for setting in sorted(expected):
        ranking = decode_rankings(frame["rankings"][setting], gold)
        rows = retrieval_metrics(ranking, gold, groups, formal=formal)
        if setting in ("final", "epochs40"):
            metrics.update(rows)
        if family == "pml_budget":
            metrics.update({setting + "_" + key: value for key, value in rows.items()})
    return metrics


def score_predictions(raw_output, *, family, parser_file):
    frame = read_prediction_frame(raw_output)
    gold, formal = protected_labels(family, parser_file)
    if family == "crowd":
        metrics = _crowd(frame, gold, formal)
    elif family == "pointnext":
        metrics = _pointnext(frame, gold, formal)
    elif family in ("pml_budget", "pml_taxonomy"):
        metrics = _pml(frame, gold, family, formal)
    else:
        raise ValueError("unknown prediction family")
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite or empty derived metrics")
    return {key: float(f"{value:.8f}") for key, value in metrics.items()}

