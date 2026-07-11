"""Trusted helpers for the pinned OpenOOD CIFAR-10 OOD protocol.

Final verification is inference-only. The complete image archive and the frozen
ResNet-18 checkpoint are built before image publication and authenticated by
SHA256 at verifier runtime. There is deliberately no cache-miss training path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch


PROTOCOL = "openood_cifar10_resnet18_full_v1"
EXPECTED_DATA_SHA256 = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
EXPECTED_MODEL_SHA256 = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"
BATCH_SIZE = 128
BASE_FORWARD_IMAGES = 106_032
BASE_FORWARD_BATCHES = 832
INVENTORY = {
    "train_images": (50_000, 3, 32, 32),
    "train_labels": (50_000,),
    "id_images": (10_000, 3, 32, 32),
    "id_labels": (10_000,),
    "ood_svhn_images": (26_032, 3, 32, 32),
    "ood_cifar100_images": (10_000, 3, 32, 32),
    "ood_tin_images": (10_000, 3, 32, 32),
}
OOD_SPLITS = (
    ("svhn", "ood_svhn_images", 26_032, 204),
    ("cifar100", "ood_cifar100_images", 10_000, 79),
    ("tin", "ood_tin_images", 10_000, 79),
)


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_surface(path: str, symbol: str):
    """Import one candidate-controlled symbol without fallback behavior."""
    spec = importlib.util.spec_from_file_location("ood_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solution surface from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"solution {path} is missing {symbol!r}")
    surface = getattr(module, symbol)
    if not callable(surface):
        raise TypeError(f"solution attribute {symbol!r} must be callable")
    return surface


def require_finite_tensor(value, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() == 0:
        raise RuntimeError(f"{label} must be a non-empty tensor")
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def require_finite_module(module: torch.nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        require_finite_tensor(parameter, f"{label} parameter {name!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} digest mismatch: got {observed}, expected {expected}")
    return observed


def load_inventory(path: Path) -> dict[str, np.ndarray]:
    """Load and validate every array in the authenticated full-image archive."""
    with np.load(path, allow_pickle=False) as archive:
        required = set(INVENTORY) | {
            "schema_version",
            "protocol",
            "cifar10_png_manifest_sha256",
            "svhn_source_sha256",
            "cifar100_source_sha256",
            "tiny_imagenet_source_sha256",
        }
        if set(archive.files) != required:
            raise RuntimeError("full OOD data has an unexpected archive inventory")
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
        if not np.array_equal(
            np.bincount(labels, minlength=10),
            np.full(10, labels.size // 10, dtype=np.int64),
        ):
            raise RuntimeError(f"{name} is not the canonical balanced CIFAR-10 split")
    return output


def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """AUROC with ID as the positive class and higher score meaning more ID-like."""
    from sklearn.metrics import roc_auc_score

    scores_id = np.asarray(scores_id, dtype=np.float64).reshape(-1)
    scores_ood = np.asarray(scores_ood, dtype=np.float64).reshape(-1)
    if scores_id.size == 0 or scores_ood.size == 0:
        raise RuntimeError("AUROC requires non-empty ID and OOD scores")
    if not np.isfinite(scores_id).all() or not np.isfinite(scores_ood).all():
        raise RuntimeError("AUROC scores contain non-finite values")
    labels = np.concatenate((np.ones(scores_id.size), np.zeros(scores_ood.size)))
    values = np.concatenate((scores_id, scores_ood))
    result = float(roc_auc_score(labels, values))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError("AUROC is outside [0, 1]")
    return result


def fpr_at_tpr(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    tpr: float = 0.95,
) -> float:
    scores_id = np.asarray(scores_id, dtype=np.float64).reshape(-1)
    scores_ood = np.asarray(scores_ood, dtype=np.float64).reshape(-1)
    if scores_id.size == 0 or scores_ood.size == 0:
        raise RuntimeError("FPR95 requires non-empty ID and OOD scores")
    if not np.isfinite(scores_id).all() or not np.isfinite(scores_ood).all():
        raise RuntimeError("FPR95 scores contain non-finite values")
    if not math.isfinite(float(tpr)) or not 0.0 < float(tpr) <= 1.0:
        raise ValueError("TPR must be in (0, 1]")
    threshold = np.quantile(scores_id, 1.0 - tpr)
    result = float((scores_ood >= threshold).mean())
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError("FPR95 is outside [0, 1]")
    return result
