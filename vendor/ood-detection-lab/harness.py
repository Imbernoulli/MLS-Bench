#!/usr/bin/env python3
"""Full-image OpenOOD verifier for the non-logit OOD research surfaces.

Every task authenticates the same uint8 image archive and frozen CIFAR-10
ResNet-18 checkpoint, then performs real inference over all 106,032 images.
Candidate code is used only for the declared post-hoc surface. Any import,
fit, scoring, shape, finite-value, or protocol failure terminates the command;
there is no cached-model training or implementation fallback.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from openood_resnet18 import ResNet18_32x32  # noqa: E402


MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(1, 3, 1, 1)
ODIN_TEMPERATURE = 1000.0
OPEN_SCORER_TASKS = {"ood-feature-score", "ood-gradient", "ood-near-far"}
FIXED_TASKS = {
    "ood-distance-metric",
    "ood-ensemble",
    "ood-input-preproc",
    "ood-layer-select",
    "ood-normalization",
    "ood-react",
    "ood-temperature",
}
ALL_TASKS = OPEN_SCORER_TASKS | FIXED_TASKS


def setting_name(task: str, ood_name: str) -> str:
    if task == "ood-near-far":
        regime = {"svhn": "far", "cifar100": "near", "tin": "medium"}[ood_name]
        return f"ood_{regime}_{ood_name}_full"
    slug = task.removeprefix("ood-").replace("-", "_")
    return f"ood_{slug}_{ood_name}_full"


def require_score_vector(value, expected: int, label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    output = np.asarray(value, dtype=np.float64)
    if output.shape != (expected,):
        raise RuntimeError(f"{label} score has shape {output.shape}, expected ({expected},)")
    if not np.isfinite(output).all():
        raise RuntimeError(f"{label} score contains non-finite values")
    return output


def require_choice(value, *, label: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}")
    return value


def require_float(value, *, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    output = float(value)
    if not math.isfinite(output) or not low <= output <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return output


@torch.inference_mode()
def extract_representations(
    model: ResNet18_32x32,
    images: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    started = time.perf_counter()
    mean = MEAN.to(device)
    std = STD.to(device)
    logits_output: list[torch.Tensor] = []
    feature_output: list[torch.Tensor] = []
    early_output: list[torch.Tensor] = []
    batches = 0
    for start in range(0, images.shape[0], common.BATCH_SIZE):
        value = torch.from_numpy(images[start:start + common.BATCH_SIZE]).to(device)
        value = (value.float().div_(255.0) - mean) / std
        early, features = model.representations(value)
        logits = model.fc(features)
        common.require_finite_tensor(early, "frozen classifier early features")
        common.require_finite_tensor(features, "frozen classifier penultimate features")
        common.require_finite_tensor(logits, "frozen classifier logits")
        if early.shape[1:] != (128,) or features.shape[1:] != (512,) or logits.shape[1:] != (10,):
            raise RuntimeError("frozen classifier emitted an unexpected representation shape")
        early_output.append(early.cpu())
        feature_output.append(features.cpu())
        logits_output.append(logits.cpu())
        batches += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    logits = torch.cat(logits_output)
    features = torch.cat(feature_output)
    early = torch.cat(early_output)
    if logits.shape != (images.shape[0], 10):
        raise RuntimeError("classifier inference did not cover the complete image split")
    return logits, features, early, batches, elapsed


def l2_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = value.double()
    norms = value.norm(dim=-1, keepdim=True)
    common.require_finite_tensor(norms, "feature norms")
    return value / norms.clamp_min(eps)


def knn_scores(
    bank: torch.Tensor,
    query: torch.Tensor,
    *,
    metric: str,
    device: torch.device,
    neighbors: int = 50,
) -> np.ndarray:
    """Chunked GPU k-NN over the complete 50k ID feature bank."""
    if bank.ndim != 2 or query.ndim != 2 or bank.shape[1] != query.shape[1]:
        raise ValueError("k-NN bank and query must have matching rank-2 feature shapes")
    if bank.shape[0] < neighbors:
        raise ValueError("k-NN fit bank is smaller than the fixed neighbour count")
    bank_device = bank.float().to(device)
    if metric == "cosine":
        bank_device = F.normalize(bank_device, dim=1)
        bank_norm = None
    elif metric == "euclidean":
        bank_norm = bank_device.square().sum(dim=1).view(1, -1)
    else:
        raise ValueError(f"unsupported k-NN metric: {metric}")

    output = torch.empty(query.shape[0], dtype=torch.float64)
    for start in range(0, query.shape[0], 256):
        value = query[start:start + 256].float().to(device)
        if metric == "cosine":
            value = F.normalize(value, dim=1)
            similarity = value @ bank_device.T
            score = similarity.topk(neighbors, dim=1, largest=True).values[:, -1]
        else:
            distance = (
                value.square().sum(dim=1, keepdim=True)
                + bank_norm
                - 2.0 * (value @ bank_device.T)
            ).clamp_min_(0.0)
            score = -distance.topk(neighbors, dim=1, largest=False).values[:, -1]
        common.require_finite_tensor(score, "k-NN score")
        output[start:start + value.shape[0]] = score.double().cpu()
    return require_score_vector(output, query.shape[0], f"{metric} k-NN")


def mahalanobis_fit(
    bank: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bank.ndim != 2 or labels.ndim != 1 or bank.shape[0] != labels.shape[0]:
        raise ValueError("Mahalanobis fit inputs have incompatible shapes")
    bank = bank.double()
    means = torch.empty(num_classes, bank.shape[1], dtype=torch.float64)
    centered = torch.empty_like(bank)
    for class_index in range(num_classes):
        index = (labels == class_index).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            raise ValueError(f"Mahalanobis fit set has no class {class_index}")
        means[class_index] = bank[index].mean(dim=0)
        centered[index] = bank[index] - means[class_index]
    covariance = centered.T @ centered / bank.shape[0]
    covariance += 1e-6 * torch.eye(bank.shape[1], dtype=torch.float64)
    precision = torch.from_numpy(np.linalg.inv(covariance.numpy()))
    common.require_finite_tensor(means, "Mahalanobis class means")
    common.require_finite_tensor(precision, "Mahalanobis precision")
    return means, precision


def mahalanobis_scores(
    query: torch.Tensor,
    means: torch.Tensor,
    precision: torch.Tensor,
) -> np.ndarray:
    if query.ndim != 2 or query.shape[1] != means.shape[1]:
        raise ValueError("Mahalanobis query has the wrong feature shape")
    output = torch.empty(query.shape[0], dtype=torch.float64)
    for start in range(0, query.shape[0], 512):
        difference = query[start:start + 512].double().unsqueeze(1) - means.unsqueeze(0)
        distance = torch.einsum("nkd,de,nke->nk", difference, precision, difference)
        common.require_finite_tensor(distance, "Mahalanobis distances")
        output[start:start + difference.shape[0]] = -distance.min(dim=1).values
    return require_score_vector(output, query.shape[0], "Mahalanobis")


def gradient_representation(logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Per-class final-layer gradient L1 contributions, shape ``[N, 10]``."""
    probabilities = torch.softmax(logits.double(), dim=1)
    difference = (probabilities - 0.1).abs()
    representation = difference * features.double().abs().sum(dim=1, keepdim=True)
    common.require_finite_tensor(representation, "gradient representation")
    return representation


def private_orders(size: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.random.default_rng(int.from_bytes(os.urandom(16), "big")).permutation(size)
    while True:
        second = np.random.default_rng(int.from_bytes(os.urandom(16), "big")).permutation(size)
        if not np.array_equal(first, second):
            return first, second


def score_profile(value: np.ndarray) -> np.ndarray:
    spread = float(np.ptp(value))
    if spread == 0.0:
        return np.zeros_like(value)
    return (value - value.min()) / spread


def permutation_checked_scores(scorer, values: tuple[torch.Tensor, ...]) -> np.ndarray:
    size = values[0].shape[0]
    if any(value.shape[0] != size for value in values):
        raise RuntimeError("candidate score inputs have inconsistent row counts")
    canonical_outputs: list[np.ndarray] = []
    for index, order in enumerate(private_orders(size), start=1):
        row_index = torch.from_numpy(order)
        observed = require_score_vector(
            scorer.score(*(value[row_index] for value in values)),
            size,
            f"private permutation {index}",
        )
        canonical = np.empty_like(observed)
        canonical[order] = observed
        canonical_outputs.append(canonical)
    first, second = canonical_outputs
    if not np.allclose(first, second, rtol=1e-5, atol=1e-7) or not np.allclose(
        score_profile(first), score_profile(second), rtol=1e-5, atol=1e-7,
    ):
        raise RuntimeError("candidate score depends on verifier-private input position")
    return first


def load_choice(solution: Path, symbol: str):
    return common.load_surface(str(solution), symbol)()


def fit_open_scorer(
    task: str,
    solution: Path,
    train_logits: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
):
    Scorer = common.load_surface(str(solution), "Scorer")
    scorer = Scorer()
    if not callable(getattr(scorer, "fit", None)) or not callable(getattr(scorer, "score", None)):
        raise TypeError("Scorer must define callable fit() and score() methods")
    if task == "ood-feature-score":
        context = SimpleNamespace(
            tr_feats=train_features,
            tr_labels=train_labels,
            num_classes=10,
            feat_dim=512,
        )
    elif task == "ood-gradient":
        context = SimpleNamespace(
            tr_gradients=gradient_representation(train_logits, train_features),
            num_classes=10,
            gradient_dim=10,
        )
    else:
        context = SimpleNamespace(
            tr_logits=train_logits,
            tr_feats=train_features,
            tr_labels=train_labels,
            num_classes=10,
            feat_dim=512,
        )
    scorer.fit(context)
    return scorer


def open_scores(
    task: str,
    scorer,
    id_values: tuple[torch.Tensor, torch.Tensor],
    ood_values: tuple[torch.Tensor, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    id_logits, id_features = id_values
    ood_logits, ood_features = ood_values
    if task == "ood-feature-score":
        values = (torch.cat((id_features, ood_features)),)
    elif task == "ood-gradient":
        values = (
            torch.cat((
                gradient_representation(id_logits, id_features),
                gradient_representation(ood_logits, ood_features),
            )),
        )
    else:
        values = (
            torch.cat((id_logits, ood_logits)),
            torch.cat((id_features, ood_features)),
        )
    scores = permutation_checked_scores(scorer, values)
    return scores[:id_logits.shape[0]], scores[id_logits.shape[0]:]


def odin_energy_scores(
    model: ResNet18_32x32,
    images: np.ndarray,
    device: torch.device,
    epsilon: float,
) -> tuple[np.ndarray, int, int, float]:
    """Run the fixed two-forward ODIN path even when epsilon is zero."""
    started = time.perf_counter()
    mean = MEAN.to(device)
    std = STD.to(device)
    output: list[torch.Tensor] = []
    batches = 0
    for start in range(0, images.shape[0], common.BATCH_SIZE):
        value = torch.from_numpy(images[start:start + common.BATCH_SIZE]).to(device)
        value = ((value.float().div_(255.0) - mean) / std).requires_grad_(True)
        logits = model(value)
        common.require_finite_tensor(logits, "ODIN initial logits")
        scaled_logits = logits / ODIN_TEMPERATURE
        target = scaled_logits.argmax(dim=1)
        loss = F.cross_entropy(scaled_logits, target)
        gradient = torch.autograd.grad(loss, value, only_inputs=True)[0]
        common.require_finite_tensor(gradient, "ODIN input gradient")
        perturbed = (value - epsilon * gradient.sign()).detach()
        with torch.no_grad():
            perturbed_logits = model(perturbed)
        common.require_finite_tensor(perturbed_logits, "ODIN perturbed logits")
        output.append((
            ODIN_TEMPERATURE
            * torch.logsumexp(perturbed_logits.double() / ODIN_TEMPERATURE, dim=1)
        ).cpu())
        batches += 2
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    scores = require_score_vector(torch.cat(output), images.shape[0], "ODIN energy")
    return scores, images.shape[0] * 2, batches, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(ALL_TASKS))
    parser.add_argument("--solution", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed != 42:
        raise RuntimeError("the pinned full OOD protocol requires seed 42")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the pinned full OOD protocol requires exactly one visible CUDA GPU")
    common.set_seeds(args.seed)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    device = torch.device("cuda:0")
    data_sha = common.require_digest(args.data, common.EXPECTED_DATA_SHA256, "full OOD data")
    model_sha = common.require_digest(
        args.checkpoint, common.EXPECTED_MODEL_SHA256, "frozen OpenOOD classifier",
    )
    inventory = common.load_inventory(args.data)

    print(
        f"OOD_PROTOCOL protocol={common.PROTOCOL} task={args.task} "
        f"model=openood_resnet18_32x32 batch_size={common.BATCH_SIZE} "
        f"seed={args.seed} status=ok",
        flush=True,
    )
    model = ResNet18_32x32().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("frozen classifier checkpoint is not a state dictionary")
    model.load_state_dict(state, strict=True)
    common.require_finite_module(model, "frozen OpenOOD classifier")
    model.eval()

    extracted: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]] = {}
    for array_name in ("train_images", "id_images", *(item[1] for item in common.OOD_SPLITS)):
        extracted[array_name] = extract_representations(model, inventory[array_name], device)
    train_logits, train_features, train_early, train_batches, train_seconds = extracted["train_images"]
    id_logits, id_features, id_early, id_batches, id_seconds = extracted["id_images"]
    id_labels = torch.from_numpy(inventory["id_labels"])
    train_labels = torch.from_numpy(inventory["train_labels"])
    id_accuracy = float(id_logits.argmax(1).eq(id_labels).double().mean().item())
    if not math.isfinite(id_accuracy) or not 0.90 <= id_accuracy <= 1.0:
        raise RuntimeError(f"frozen classifier accuracy sanity check failed: {id_accuracy:.6f}")
    base_images = sum(inventory[name].shape[0] for name in common.INVENTORY if name.endswith("images"))
    base_batches = train_batches + id_batches + sum(extracted[item[1]][3] for item in common.OOD_SPLITS)
    if base_images != common.BASE_FORWARD_IMAGES or base_batches != common.BASE_FORWARD_BATCHES:
        raise RuntimeError("base full-image forward inventory is incomplete")
    print(
        f"OOD_CLASSIFIER task={args.task} id_acc={id_accuracy:.8f} n_train=50000 "
        f"n_id=10000 train_batches={train_batches} id_batches={id_batches} "
        f"train_seconds={train_seconds:.6f} id_seconds={id_seconds:.6f} status=ok",
        flush=True,
    )

    scorer = fit_open_scorer(
        args.task, args.solution, train_logits, train_features, train_labels,
    ) if args.task in OPEN_SCORER_TASKS else None
    fixed_state: dict[str, object] = {}
    if args.task == "ood-temperature":
        fixed_state["temperature"] = require_float(
            load_choice(args.solution, "select_temperature"),
            label="temperature", low=1e-3, high=1e4,
        )
    elif args.task == "ood-distance-metric":
        fixed_state["metric"] = require_choice(
            load_choice(args.solution, "select_distance_metric"),
            label="distance metric", choices={"cosine", "euclidean"},
        )
    elif args.task == "ood-normalization":
        normalization = require_choice(
            load_choice(args.solution, "select_feature_normalization"),
            label="feature normalization", choices={"l2", "raw"},
        )
        transform = l2_normalize if normalization == "l2" else lambda value: value.double()
        fixed_state["transform"] = transform
        fixed_state["maha"] = mahalanobis_fit(transform(train_features), train_labels, 10)
    elif args.task == "ood-layer-select":
        layers = require_choice(
            load_choice(args.solution, "select_feature_layers"),
            label="feature layers", choices={"concat", "penultimate"},
        )
        fixed_state["layers"] = layers
        train_selected = train_features if layers == "penultimate" else torch.cat((train_early, train_features), 1)
        train_selected = l2_normalize(train_selected)
        fixed_state["maha"] = mahalanobis_fit(train_selected, train_labels, 10)
    elif args.task == "ood-ensemble":
        fixed_state["ensemble"] = require_choice(
            load_choice(args.solution, "select_ensemble"),
            label="ensemble", choices={"energy_knn", "knn"},
        )
        train_knn = knn_scores(
            train_features, train_features, metric="euclidean", device=device,
            neighbors=51,
        )
        fixed_state["train_knn_mean"] = float(train_knn.mean())
        fixed_state["train_knn_std"] = float(train_knn.std())
        train_energy = torch.logsumexp(train_logits.double(), dim=1).numpy()
        fixed_state["train_energy_mean"] = float(train_energy.mean())
        fixed_state["train_energy_std"] = float(train_energy.std())
        if min(fixed_state["train_knn_std"], fixed_state["train_energy_std"]) <= 0.0:
            raise RuntimeError("ensemble fit statistic has zero variance")
    elif args.task == "ood-react":
        quantile = load_choice(args.solution, "select_clip_quantile")
        fixed_state["clip"] = None if quantile is None else torch.quantile(
            train_features.double().flatten(),
            require_float(quantile, label="clip quantile", low=0.5, high=1.0),
        )
    elif args.task == "ood-input-preproc":
        fixed_state["epsilon"] = require_float(
            load_choice(args.solution, "select_preprocess_epsilon"),
            label="preprocess epsilon", low=0.0, high=0.1,
        )

    print(f"SCORER_FIT task={args.task} n_fit=50000 status=ok", flush=True)
    id_score_batches = 0
    id_extra_images = 0
    if args.task == "ood-input-preproc":
        id_scores, id_extra_images, id_score_batches, _id_score_seconds = odin_energy_scores(
            model, inventory["id_images"], device, fixed_state["epsilon"],
        )
    else:
        id_scores = None

    total_extra_images = id_extra_images
    total_extra_batches = id_score_batches
    for ood_name, array_name, expected_count, expected_base_batches in common.OOD_SPLITS:
        started = time.perf_counter()
        ood_logits, ood_features, ood_early, ood_batches, base_seconds = extracted[array_name]
        if ood_batches != expected_base_batches or ood_logits.shape[0] != expected_count:
            raise RuntimeError(f"{ood_name} base inference inventory is incomplete")
        ood_score_batches = 0
        ood_extra_images = 0

        if args.task in OPEN_SCORER_TASKS:
            score_id, score_ood = open_scores(
                args.task, scorer, (id_logits, id_features), (ood_logits, ood_features),
            )
        elif args.task == "ood-temperature":
            temperature = fixed_state["temperature"]
            score_id = (temperature * torch.logsumexp(id_logits.double() / temperature, 1)).numpy()
            score_ood = (temperature * torch.logsumexp(ood_logits.double() / temperature, 1)).numpy()
        elif args.task == "ood-distance-metric":
            metric = fixed_state["metric"]
            score_id = knn_scores(train_features, id_features, metric=metric, device=device)
            score_ood = knn_scores(train_features, ood_features, metric=metric, device=device)
        elif args.task in {"ood-normalization", "ood-layer-select"}:
            means, precision = fixed_state["maha"]
            if args.task == "ood-normalization":
                transform = fixed_state["transform"]
                selected_id = transform(id_features)
                selected_ood = transform(ood_features)
            else:
                selected_id = id_features if fixed_state["layers"] == "penultimate" else torch.cat((id_early, id_features), 1)
                selected_ood = ood_features if fixed_state["layers"] == "penultimate" else torch.cat((ood_early, ood_features), 1)
                selected_id = l2_normalize(selected_id)
                selected_ood = l2_normalize(selected_ood)
            score_id = mahalanobis_scores(selected_id, means, precision)
            score_ood = mahalanobis_scores(selected_ood, means, precision)
        elif args.task == "ood-ensemble":
            id_knn = knn_scores(train_features, id_features, metric="euclidean", device=device)
            ood_knn = knn_scores(train_features, ood_features, metric="euclidean", device=device)
            if fixed_state["ensemble"] == "knn":
                score_id, score_ood = id_knn, ood_knn
            else:
                id_energy = torch.logsumexp(id_logits.double(), 1).numpy()
                ood_energy = torch.logsumexp(ood_logits.double(), 1).numpy()
                score_id = (
                    (id_knn - fixed_state["train_knn_mean"]) / fixed_state["train_knn_std"]
                    + (id_energy - fixed_state["train_energy_mean"]) / fixed_state["train_energy_std"]
                )
                score_ood = (
                    (ood_knn - fixed_state["train_knn_mean"]) / fixed_state["train_knn_std"]
                    + (ood_energy - fixed_state["train_energy_mean"]) / fixed_state["train_energy_std"]
                )
        elif args.task == "ood-react":
            clip = fixed_state["clip"]
            weight = model.fc.weight.detach().cpu().double()
            bias = model.fc.bias.detach().cpu().double()

            def react_score(features: torch.Tensor) -> np.ndarray:
                value = features.double() if clip is None else torch.minimum(features.double(), clip)
                logits = value @ weight.T + bias
                common.require_finite_tensor(logits, "ReAct reconstructed logits")
                return torch.logsumexp(logits, 1).numpy()

            score_id, score_ood = react_score(id_features), react_score(ood_features)
        elif args.task == "ood-input-preproc":
            score_id = id_scores
            score_ood, ood_extra_images, ood_score_batches, _score_seconds = odin_energy_scores(
                model, inventory[array_name], device, fixed_state["epsilon"],
            )
            total_extra_images += ood_extra_images
            total_extra_batches += ood_score_batches
        else:
            raise AssertionError(f"unhandled OOD task {args.task}")

        score_id = require_score_vector(score_id, 10_000, f"{ood_name} ID")
        score_ood = require_score_vector(score_ood, expected_count, f"{ood_name} OOD")
        metric_auroc = common.auroc(score_id, score_ood)
        metric_fpr95 = common.fpr_at_tpr(score_id, score_ood, tpr=0.95)
        elapsed = base_seconds + (time.perf_counter() - started)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError(f"{ood_name} evaluation runtime is invalid")
        print(
            f"OOD_METRICS protocol={common.PROTOCOL} task={args.task} "
            f"setting={setting_name(args.task, ood_name)} ood={ood_name} "
            f"auroc={metric_auroc:.8f} fpr95={metric_fpr95:.8f} id_acc={id_accuracy:.8f} "
            f"n_fit=50000 n_id=10000 n_ood={expected_count} "
            f"base_ood_batches={ood_batches} id_score_batches={id_score_batches} "
            f"ood_score_batches={ood_score_batches} inference_seconds={elapsed:.6f} status=ok",
            flush=True,
        )

    task_images = base_images + total_extra_images
    task_batches = base_batches + total_extra_batches
    expected_task_images = 218_096 if args.task == "ood-input-preproc" else 106_032
    expected_task_batches = 1_714 if args.task == "ood-input-preproc" else 832
    if task_images != expected_task_images or task_batches != expected_task_batches:
        raise RuntimeError(
            f"task forward inventory mismatch: images={task_images}, batches={task_batches}"
        )
    print(
        f"OOD_COMPLETE protocol={common.PROTOCOL} task={args.task} "
        f"data_sha256={data_sha} checkpoint_sha256={model_sha} "
        f"n_fit=50000 n_id=10000 n_svhn=26032 n_cifar100=10000 n_tin=10000 "
        f"base_forward_images={base_images} base_forward_batches={base_batches} "
        f"task_forward_images={task_images} task_forward_batches={task_batches} status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
