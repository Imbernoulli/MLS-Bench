#!/usr/bin/env python3
"""Evaluate reproducible logit-only OOD anchor candidates from a staged dump."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import common  # noqa: E402


PROTOCOL = "openood_cifar10_resnet18_full_v1"
DATA_SHA256 = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
CHECKPOINT_SHA256 = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"
EXPECTED_SHAPES = {
    "train_logits": (50_000, 10),
    "id_logits": (10_000, 10),
    "svhn_logits": (26_032, 10),
    "cifar100_logits": (10_000, 10),
    "tin_logits": (10_000, 10),
}
OOD_SPLITS = ("svhn", "cifar100", "tin")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dump(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["protocol"].item()) != PROTOCOL:
            raise RuntimeError("logit dump has the wrong protocol")
        if str(archive["data_sha256"].item()) != DATA_SHA256:
            raise RuntimeError("logit dump has the wrong data digest")
        if str(archive["checkpoint_sha256"].item()) != CHECKPOINT_SHA256:
            raise RuntimeError("logit dump has the wrong checkpoint digest")
        if int(archive["total_forward_images"].item()) != 106_032:
            raise RuntimeError("logit dump has the wrong image count")
        if int(archive["total_forward_batches"].item()) != 832:
            raise RuntimeError("logit dump has the wrong batch count")
        id_accuracy = float(archive["id_accuracy"].item())
        if not math.isfinite(id_accuracy) or not 0.90 <= id_accuracy <= 1.0:
            raise RuntimeError("logit dump has an invalid classifier accuracy")
        arrays = {name: archive[name].astype(np.float64) for name in EXPECTED_SHAPES}
    for name, shape in EXPECTED_SHAPES.items():
        value = arrays[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"{name} has invalid shape or values")
    arrays["id_accuracy"] = np.asarray(id_accuracy)
    return arrays


def softmax(value: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = value / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    output = np.exp(scaled)
    output /= output.sum(axis=1, keepdims=True)
    return output


def logsumexp(value: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = value / temperature
    maximum = scaled.max(axis=1)
    return temperature * (
        maximum + np.log(np.exp(scaled - maximum[:, None]).sum(axis=1))
    )


def quadratic_distances(
    value: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    output = np.empty((value.shape[0], means.shape[0]), dtype=np.float64)
    for start in range(0, value.shape[0], 8192):
        block = value[start:start + 8192]
        difference = block[:, None, :] - means[None, :, :]
        output[start:start + block.shape[0]] = np.einsum(
            "nci,ij,ncj->nc", difference, precision, difference, optimize=True,
        )
    return output


def add_fixed_candidates(
    scores: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
) -> None:
    names = tuple(EXPECTED_SHAPES)
    for temperature in (0.5, 1.0, 2.0, 10.0, 100.0, 1000.0):
        label = f"msp_t{temperature:g}"
        scores[label] = {
            name: softmax(arrays[name], temperature).max(axis=1) for name in names
        }
        label = f"energy_t{temperature:g}"
        scores[label] = {
            name: logsumexp(arrays[name], temperature) for name in names
        }
    scores["max_logit"] = {name: arrays[name].max(axis=1) for name in names}
    scores["max_centered_logit"] = {
        name: (arrays[name] - arrays[name].mean(axis=1, keepdims=True)).max(axis=1)
        for name in names
    }
    scores["logit_l2_centered"] = {
        name: np.linalg.norm(
            arrays[name] - arrays[name].mean(axis=1, keepdims=True), axis=1,
        )
        for name in names
    }
    scores["top1_margin"] = {
        name: np.partition(arrays[name], -2, axis=1)[:, -1]
        - np.partition(arrays[name], -2, axis=1)[:, -2]
        for name in names
    }
    scores["negative_entropy"] = {}
    for name in names:
        probability = softmax(arrays[name])
        scores["negative_entropy"][name] = np.sum(
            probability * np.log(np.maximum(probability, 1e-300)), axis=1,
        )


def add_fitted_candidates(
    scores: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
) -> None:
    names = tuple(EXPECTED_SHAPES)
    train = arrays["train_logits"]
    pseudo_label = train.argmax(axis=1)
    counts = np.bincount(pseudo_label, minlength=10)
    if np.any(counts < 100):
        raise RuntimeError(f"pseudo-class inventory is invalid: {counts.tolist()}")
    means = np.stack([train[pseudo_label == label].mean(axis=0) for label in range(10)])
    residual = train - means[pseudo_label]
    covariance = residual.T @ residual / max(1, residual.shape[0] - 10)
    global_mean = train.mean(axis=0)
    global_residual = train - global_mean
    global_covariance = global_residual.T @ global_residual / (train.shape[0] - 1)
    covariance_scale = float(np.trace(covariance) / covariance.shape[0])
    global_scale = float(np.trace(global_covariance) / global_covariance.shape[0])

    for ridge in (1e-6, 1e-3, 1e-1):
        precision = np.linalg.inv(
            covariance + np.eye(covariance.shape[0]) * covariance_scale * ridge
        )
        global_precision = np.linalg.inv(
            global_covariance
            + np.eye(global_covariance.shape[0]) * global_scale * ridge
        )
        minimum: dict[str, np.ndarray] = {}
        predicted: dict[str, np.ndarray] = {}
        relative: dict[str, np.ndarray] = {}
        for name in names:
            value = arrays[name]
            class_distance = quadratic_distances(value, means, precision)
            minimum[name] = -class_distance.min(axis=1)
            predicted[name] = -class_distance[
                np.arange(value.shape[0]), value.argmax(axis=1)
            ]
            difference = value - global_mean
            global_distance = np.einsum(
                "ni,ij,nj->n", difference, global_precision, difference, optimize=True,
            )
            relative[name] = global_distance - class_distance.min(axis=1)
        scores[f"pseudo_maha_min_r{ridge:g}"] = minimum
        scores[f"pseudo_maha_pred_r{ridge:g}"] = predicted
        scores[f"pseudo_rmd_r{ridge:g}"] = relative

    variance = np.stack(
        [
            residual[pseudo_label == label].var(axis=0, ddof=1) + 1e-6
            for label in range(10)
        ]
    )
    scores["pseudo_diag_gaussian"] = {}
    scores["pseudo_cosine"] = {}
    normalized_means = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    for name in names:
        value = arrays[name]
        difference = value[:, None, :] - means[None, :, :]
        diagonal_nll = np.sum(
            difference * difference / variance[None, :, :]
            + np.log(variance[None, :, :]),
            axis=2,
        )
        scores["pseudo_diag_gaussian"][name] = -diagonal_nll.min(axis=1)
        normalized_value = value / np.maximum(
            np.linalg.norm(value, axis=1, keepdims=True), 1e-12,
        )
        scores["pseudo_cosine"][name] = (
            normalized_value @ normalized_means.T
        ).max(axis=1)


def add_id_normalized_combinations(
    scores: dict[str, dict[str, np.ndarray]],
) -> None:
    base_names = (
        "msp_t1",
        "max_logit",
        "max_centered_logit",
        "negative_entropy",
        "pseudo_rmd_r0.001",
        "pseudo_maha_min_r0.001",
        "pseudo_diag_gaussian",
        "pseudo_cosine",
    )
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for name in base_names:
        train_score = scores[name]["train_logits"]
        center = float(train_score.mean())
        scale = float(train_score.std())
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"candidate {name} cannot be normalized from ID fit data")
        normalized[name] = {
            split: (value - center) / scale for split, value in scores[name].items()
        }
    for second in base_names[1:]:
        for weight in (0.1, 0.25, 0.5, 1.0, 2.0):
            name = f"id_z_msp_plus_{weight:g}_{second}"
            scores[name] = {
                split: normalized["msp_t1"][split]
                + weight * normalized[second][split]
                for split in normalized["msp_t1"]
            }


def evaluate(
    scores: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, float | str]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for candidate, values in scores.items():
        if any(
            value.shape != (EXPECTED_SHAPES[name][0],) or not np.isfinite(value).all()
            for name, value in values.items()
        ):
            raise RuntimeError(f"candidate {candidate} emitted invalid scores")
        results[candidate] = {}
        for split in OOD_SPLITS:
            results[candidate][split] = {
                "auroc": common.auroc(values["id_logits"], values[f"{split}_logits"]),
                "fpr95": common.fpr_at_tpr(
                    values["id_logits"], values[f"{split}_logits"], tpr=0.95,
                ),
            }
    msp = results["msp_t1"]
    ranking = []
    for candidate, candidate_results in results.items():
        improvements = [
            candidate_results[split]["auroc"] - msp[split]["auroc"]
            for split in OOD_SPLITS
        ]
        ranking.append(
            {
                "candidate": candidate,
                "worst_auroc_improvement_over_msp": min(improvements),
                "mean_auroc_improvement_over_msp": float(np.mean(improvements)),
                "mean_auroc": float(
                    np.mean([candidate_results[split]["auroc"] for split in OOD_SPLITS])
                ),
            }
        )
    ranking.sort(
        key=lambda row: (
            row["worst_auroc_improvement_over_msp"],
            row["mean_auroc_improvement_over_msp"],
        ),
        reverse=True,
    )
    return results, ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arrays = load_dump(args.dump)
    scores: dict[str, dict[str, np.ndarray]] = {}
    add_fixed_candidates(scores, arrays)
    add_fitted_candidates(scores, arrays)
    add_id_normalized_combinations(scores)
    results, ranking = evaluate(scores)
    payload = {
        "protocol": PROTOCOL,
        "dump_sha256": sha256(args.dump),
        "data_sha256": DATA_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "id_accuracy": float(arrays["id_accuracy"]),
        "candidate_count": len(results),
        "results": results,
        "ranking": ranking,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in ranking[:30]:
        candidate = str(row["candidate"])
        fields = []
        for split in OOD_SPLITS:
            metric = results[candidate][split]
            fields.append(
                f"{split}:auroc={metric['auroc']:.8f},fpr95={metric['fpr95']:.8f}"
            )
        print(
            f"OOD_CANDIDATE name={candidate} "
            f"worst_delta={row['worst_auroc_improvement_over_msp']:.8f} "
            f"mean_delta={row['mean_auroc_improvement_over_msp']:.8f} "
            + " ".join(fields),
            flush=True,
        )


if __name__ == "__main__":
    main()
