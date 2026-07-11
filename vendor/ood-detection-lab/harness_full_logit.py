#!/usr/bin/env python3
"""Full-image OpenOOD-style CIFAR-10 logit-score evaluation.

The frozen classifier performs a real forward pass over every image in the
canonical CIFAR-10 train/test, SVHN test, CIFAR-100 test, and Tiny-ImageNet val
inventories.  Only the resulting logits are exposed to the editable scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from openood_resnet18 import ResNet18_32x32  # noqa: E402


PROTOCOL = "openood_cifar10_resnet18_full_v1"
EXPECTED_DATA_SHA256 = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
EXPECTED_MODEL_SHA256 = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"
BATCH_SIZE = 128
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(1, 3, 1, 1)
INVENTORY = {
    "train_images": (50_000, 3, 32, 32),
    "train_labels": (50_000,),
    "id_images": (10_000, 3, 32, 32),
    "id_labels": (10_000,),
    "ood_svhn_images": (26_032, 3, 32, 32),
    "ood_cifar100_images": (10_000, 3, 32, 32),
    "ood_tin_images": (10_000, 3, 32, 32),
}
OOD_SETTINGS = (
    ("ood_logit_svhn_full", "svhn", "ood_svhn_images"),
    ("ood_logit_cifar100_full", "cifar100", "ood_cifar100_images"),
    ("ood_logit_tin_full", "tin", "ood_tin_images"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} digest mismatch: got {observed}, expected {expected}")
    return observed


def load_inventory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["schema_version"].item()) != 1:
            raise RuntimeError("full OOD data has the wrong schema version")
        if str(archive["protocol"].item()) != PROTOCOL:
            raise RuntimeError("full OOD data has the wrong protocol identifier")
        output = {name: archive[name] for name in INVENTORY}
    for name, expected_shape in INVENTORY.items():
        value = output[name]
        expected_dtype = np.int64 if name.endswith("labels") else np.uint8
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise RuntimeError(
                f"{name} has shape/dtype {value.shape}/{value.dtype}, "
                f"expected {expected_shape}/{expected_dtype}"
            )
    for name in ("train_labels", "id_labels"):
        labels = output[name]
        if labels.min() != 0 or labels.max() != 9:
            raise RuntimeError(f"{name} contains an invalid CIFAR-10 class")
    return output


@torch.inference_mode()
def extract_logits(
    model: torch.nn.Module,
    images: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, int, float]:
    started = time.perf_counter()
    mean = MEAN.to(device)
    std = STD.to(device)
    batches: list[torch.Tensor] = []
    count = 0
    for start in range(0, images.shape[0], BATCH_SIZE):
        value = torch.from_numpy(images[start:start + BATCH_SIZE]).to(device)
        value = value.float().div_(255.0)
        value = (value - mean) / std
        logits = model(value)
        common.require_finite_tensor(logits, "frozen classifier logits")
        if logits.ndim != 2 or logits.shape[1] != 10:
            raise RuntimeError(f"classifier emitted invalid logits shape {tuple(logits.shape)}")
        batches.append(logits.cpu())
        count += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    output = torch.cat(batches, dim=0)
    if output.shape != (images.shape[0], 10):
        raise RuntimeError("classifier inference did not cover the complete image inventory")
    return output, count, elapsed


def require_score_vector(value, expected: int, label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    output = np.asarray(value, dtype=np.float64)
    if output.shape != (expected,):
        raise RuntimeError(f"{label} scorer output has shape {output.shape}, expected ({expected},)")
    if not np.isfinite(output).all():
        raise RuntimeError(f"{label} scorer output contains non-finite values")
    return output


def private_orders(size: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.random.default_rng(int.from_bytes(os.urandom(16), "big")).permutation(size)
    while True:
        second = np.random.default_rng(int.from_bytes(os.urandom(16), "big")).permutation(size)
        if not np.array_equal(first, second):
            return first, second


def permutation_checked_scores(scorer, logits: torch.Tensor) -> np.ndarray:
    canonical_outputs = []
    for index, order in enumerate(private_orders(logits.shape[0]), start=1):
        row_index = torch.from_numpy(order)
        observed = require_score_vector(
            scorer.score(logits[row_index]), logits.shape[0], f"private_permutation_{index}",
        )
        canonical = np.empty_like(observed)
        canonical[order] = observed
        canonical_outputs.append(canonical)
    first, second = canonical_outputs
    if not np.allclose(first, second, rtol=1e-5, atol=1e-7):
        raise RuntimeError("scorer output depends on verifier-private input position")
    first_range = float(np.ptp(first))
    second_range = float(np.ptp(second))
    first_profile = np.zeros_like(first) if first_range == 0 else (first - first.min()) / first_range
    second_profile = (
        np.zeros_like(second) if second_range == 0 else (second - second.min()) / second_range
    )
    if not np.allclose(first_profile, second_profile, rtol=1e-5, atol=1e-7):
        raise RuntimeError("scorer output profile depends on verifier-private input position")
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
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
    data_sha = require_digest(args.data, EXPECTED_DATA_SHA256, "full OOD data")
    model_sha = require_digest(args.checkpoint, EXPECTED_MODEL_SHA256, "frozen classifier")
    inventory = load_inventory(args.data)
    print(
        f"OOD_PROTOCOL protocol={PROTOCOL} task=ood-logit-score "
        f"model=openood_resnet18_32x32 "
        f"batch_size={BATCH_SIZE} seed={args.seed} device={torch.cuda.get_device_name(0)!r}",
        flush=True,
    )

    model = ResNet18_32x32().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise RuntimeError("frozen classifier checkpoint is not a state dictionary")
    model.load_state_dict(state, strict=True)
    common.require_finite_module(model, "frozen OpenOOD classifier")
    model.eval()

    train_logits, train_batches, train_seconds = extract_logits(
        model, inventory["train_images"], device,
    )
    id_logits, id_batches, id_seconds = extract_logits(model, inventory["id_images"], device)
    id_labels = torch.from_numpy(inventory["id_labels"])
    id_accuracy = float(id_logits.argmax(1).eq(id_labels).double().mean().item())
    if not math.isfinite(id_accuracy) or not 0.90 <= id_accuracy <= 1.0:
        raise RuntimeError(f"frozen classifier accuracy sanity check failed: {id_accuracy:.6f}")
    print(
        f"OOD_CLASSIFIER id_acc={id_accuracy:.6f} n_train={train_logits.shape[0]} "
        f"n_id={id_logits.shape[0]} train_batches={train_batches} id_batches={id_batches} "
        f"train_seconds={train_seconds:.6f} id_seconds={id_seconds:.6f}",
        flush=True,
    )

    Scorer = common.load_surface(str(args.solution), "Scorer")
    scorer = Scorer()
    if not callable(getattr(scorer, "fit", None)) or not callable(getattr(scorer, "score", None)):
        raise TypeError("Scorer must define callable fit() and score() methods")
    scorer.fit(SimpleNamespace(tr_logits=train_logits, num_classes=10))
    print("SCORER_FIT status=ok n_fit=50000", flush=True)

    total_batches = train_batches + id_batches
    total_images = train_logits.shape[0] + id_logits.shape[0]
    for setting, ood_name, array_name in OOD_SETTINGS:
        ood_logits, ood_batches, inference_seconds = extract_logits(
            model, inventory[array_name], device,
        )
        total_batches += ood_batches
        total_images += ood_logits.shape[0]
        combined = torch.cat((id_logits, ood_logits), dim=0)
        scores = permutation_checked_scores(scorer, combined)
        id_scores = scores[:id_logits.shape[0]]
        ood_scores = scores[id_logits.shape[0]:]
        auroc = common.auroc(id_scores, ood_scores)
        fpr95 = common.fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
        if not np.isfinite((auroc, fpr95, id_accuracy, inference_seconds)).all():
            raise RuntimeError(f"{setting} produced a non-finite result")
        print(
            f"OOD_METRICS protocol={PROTOCOL} task=ood-logit-score "
            f"setting={setting} ood={ood_name} "
            f"auroc={auroc:.8f} fpr95={fpr95:.8f} id_acc={id_accuracy:.8f} "
            f"n_fit=50000 n_id={id_logits.shape[0]} n_ood={ood_logits.shape[0]} "
            f"forward_batches={ood_batches} inference_seconds={inference_seconds:.6f} status=ok",
            flush=True,
        )

    if total_images != 106_032 or total_batches != 832:
        raise RuntimeError(
            f"full forward inventory mismatch: images={total_images}, batches={total_batches}"
        )
    print(
        f"OOD_COMPLETE protocol={PROTOCOL} task=ood-logit-score "
        f"data_sha256={data_sha} "
        f"checkpoint_sha256={model_sha} n_fit=50000 n_id=10000 n_svhn=26032 "
        f"n_cifar100=10000 n_tin=10000 total_forward_images={total_images} "
        f"total_forward_batches={total_batches} status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
