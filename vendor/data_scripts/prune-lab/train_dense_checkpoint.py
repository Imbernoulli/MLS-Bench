#!/usr/bin/env python3
"""Build the pinned 200-epoch CIFAR-10 ResNet-18 artifact on a GPU worker.

This operator tool performs no download or environment installation. It accepts
only an already extracted canonical CIFAR-10 Python archive, trains the fixed
dense model, re-evaluates the saved state on all 10,000 test examples, and emits
the checkpoint, its SHA-256 sidecar, and a provenance record atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

# PyTorch requires this to be present before the first CUDA context is created
# when deterministic algorithms cover CuBLAS-backed operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


PROTOCOL = "cifar10-resnet18-200ep-v1"
ARCH = "torchvision-resnet18-cifar-stem-v1"
TRAIN_EPOCHS = 200
EARLY_EPOCH = 2
SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
CHECKPOINT_NAME = "dense_resnet18_cifar10.pt"
PROVENANCE_NAME = "dense_resnet18_cifar10.provenance.json"
CIFAR_MD5 = {
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _validate_git_identity(name: str, value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized):
        raise SystemExit(f"{name} must be an exact 40-hex Git identity")
    return normalized


def _validate_cifar(data_root: Path) -> Path:
    archive = data_root.resolve() / "cifar-10-batches-py"
    if archive.is_symlink() or not archive.is_dir():
        raise SystemExit(f"canonical CIFAR-10 directory is missing: {archive}")
    for name, expected in CIFAR_MD5.items():
        path = archive / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"canonical CIFAR-10 member is missing: {path}")
        actual = _digest(path, "md5")
        if actual != expected:
            raise SystemExit(
                f"canonical CIFAR-10 member failed MD5 validation: {name} "
                f"expected={expected} actual={actual}"
            )
    return archive


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()

    source_commit = _validate_git_identity("source commit", args.source_commit)
    source_tree = _validate_git_identity("source tree", args.source_tree)
    _validate_cifar(args.data_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / CHECKPOINT_NAME
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    provenance_path = output_dir / PROVENANCE_NAME
    for path in (checkpoint, sidecar, provenance_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite an existing artifact: {path}")

    vendor_root = Path(__file__).resolve().parents[2]
    harness_root = vendor_root / "prune-lab"
    sys.path.insert(0, str(harness_root))
    import harness  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    if harness.DENSE_PROTOCOL != PROTOCOL:
        raise SystemExit("training protocol and verifier protocol disagree")
    if harness.BATCH != BATCH_SIZE or harness.WEIGHT_DECAY != WEIGHT_DECAY:
        raise SystemExit("training recipe and verifier constants disagree")
    if not torch.cuda.is_available():
        raise SystemExit("the dense checkpoint builder requires one CUDA GPU")

    harness.set_all_seeds(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    train_loader, test_loader = harness.load_cifar10(
        str(args.data_root.resolve()), BATCH_SIZE, NUM_WORKERS
    )
    model = harness.build_resnet18_cifar().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_EPOCHS
    )
    early_state: dict[str, torch.Tensor] = {}
    started = time.monotonic()
    print(
        "PRUNE_DENSE_BUILD_START "
        f"protocol={PROTOCOL} epochs={TRAIN_EPOCHS} seed={SEED} "
        f"batch={BATCH_SIZE} train={len(train_loader.dataset)} "
        f"test={len(test_loader.dataset)} source_commit={source_commit} "
        f"source_tree={source_tree} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} gpu={torch.cuda.get_device_name(0)!r}",
        flush=True,
    )

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite dense training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            count = targets.numel()
            loss_sum += float(loss.detach()) * count
            examples += count
        scheduler.step()
        if epoch == EARLY_EPOCH:
            early_state = _cpu_state_dict(model)
        if epoch == 1 or epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
            elapsed = time.monotonic() - started
            print(
                "PRUNE_DENSE_PROGRESS "
                f"epoch={epoch}/{TRAIN_EPOCHS} "
                f"loss={loss_sum / max(1, examples):.8f} "
                f"lr={optimizer.param_groups[0]['lr']:.10f} "
                f"elapsed_seconds={elapsed:.3f}",
                flush=True,
            )

    if not early_state or set(early_state) != set(model.state_dict()):
        raise RuntimeError("the epoch-2 rewind state was not captured completely")
    dense_acc = float(harness.accuracy(model, test_loader, device))
    if not math.isfinite(dense_acc) or not 0.0 <= dense_acc <= 1.0:
        raise RuntimeError("the final dense accuracy is invalid")

    artifact = {
        "state_dict": _cpu_state_dict(model),
        "early_state_dict": early_state,
        "dense_acc": dense_acc,
        "protocol": PROTOCOL,
        "train_epochs": TRAIN_EPOCHS,
        "early_epoch": EARLY_EPOCH,
        "arch": ARCH,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "sgd-momentum-0.9-nesterov",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "schedule": "cosine-200ep",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "cifar10_files_md5": CIFAR_MD5,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    temporary = output_dir / f".{CHECKPOINT_NAME}.tmp-{os.getpid()}"
    try:
        torch.save(artifact, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        verification_model = harness.build_resnet18_cifar().to(device)
        verification_model.load_state_dict(loaded["state_dict"], strict=True)
        measured_acc = float(
            harness.accuracy(verification_model, test_loader, device)
        )
        if abs(measured_acc - dense_acc) > 5e-4:
            raise RuntimeError(
                "saved checkpoint accuracy does not match the full test evaluation"
            )
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)

    checkpoint_sha256 = _digest(checkpoint)
    sidecar.write_text(checkpoint_sha256 + "\n")
    elapsed_seconds = time.monotonic() - started
    provenance = {
        "arch": ARCH,
        "batch_size": BATCH_SIZE,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "cifar10_files_md5": CIFAR_MD5,
        "dense_acc": dense_acc,
        "early_epoch": EARLY_EPOCH,
        "elapsed_seconds": elapsed_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "learning_rate": LEARNING_RATE,
        "protocol": PROTOCOL,
        "schedule": "cosine-200ep",
        "seed": SEED,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "train_epochs": TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
    }
    provenance_path.write_text(_canonical_json(provenance))
    print(
        "PRUNE_DENSE_BUILD_OK "
        f"protocol={PROTOCOL} epochs={TRAIN_EPOCHS} dense_acc={dense_acc:.6f} "
        f"checkpoint_sha256={checkpoint_sha256} "
        f"checkpoint_bytes={checkpoint.stat().st_size} "
        f"provenance_sha256={_digest(provenance_path)} "
        f"elapsed_seconds={elapsed_seconds:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
