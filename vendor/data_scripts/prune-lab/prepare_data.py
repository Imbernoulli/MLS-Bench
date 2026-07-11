#!/usr/bin/env python3
"""Validate the offline CIFAR-10/checkpoint bundle for prune-lab.

No dataset download or dense-model training is performed here. Operators stage the
canonical extracted CIFAR-10 Python archive and the pinned dense checkpoint before
building the per-repo image. This step verifies them and writes a frozen manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL = "cifar10-resnet18-200ep-v1"
# This must be populated from the independently approved checkpoint artifact.
# A digest read only from the checkpoint's own bundle is not a trust anchor.
EXPECTED_CHECKPOINT_SHA256 = ""
CIFAR_MD5 = {
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}


def _digest(path: Path, algorithm: str) -> str:
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


def _checkpoint_pin() -> str:
    expected = EXPECTED_CHECKPOINT_SHA256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit(
            "EXPECTED_CHECKPOINT_SHA256 is not configured with the independently "
            "approved dense-checkpoint digest"
        )
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    expected_checkpoint_sha256 = _checkpoint_pin()
    root = args.data_root.expanduser().resolve() / "prune-lab"
    cifar = root / "cifar" / "cifar-10-batches-py"
    for name, expected in CIFAR_MD5.items():
        path = cifar / name
        if not path.is_file() or _digest(path, "md5") != expected:
            raise SystemExit(f"canonical CIFAR-10 member failed MD5 validation: {path}")

    checkpoint = root / "dense_resnet18_cifar10.pt"
    sidecar = root / "dense_resnet18_cifar10.pt.sha256"
    if not checkpoint.is_file() or not sidecar.is_file():
        raise SystemExit("pinned dense checkpoint or SHA-256 sidecar is missing")
    expected_sha = sidecar.read_text().strip().lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise SystemExit("dense checkpoint SHA-256 sidecar is malformed")
    if expected_sha != expected_checkpoint_sha256:
        raise SystemExit("dense checkpoint sidecar does not match the repository trust anchor")
    actual_sha = _digest(checkpoint, "sha256")
    if actual_sha != expected_checkpoint_sha256:
        raise SystemExit("dense checkpoint SHA-256 does not match the repository trust anchor")

    manifest = {
        "checkpoint": "dense_resnet18_cifar10.pt",
        "checkpoint_sha256": actual_sha,
        "cifar10_files_md5": CIFAR_MD5,
        "protocol": PROTOCOL,
        "test_count": 10_000,
        "train_count": 50_000,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(_canonical_json(manifest))
    print(
        "PRUNE_PREPARE_COMPLETE "
        f"protocol={PROTOCOL} train=50000 test=10000 "
        f"manifest_sha256={_digest(manifest_path, 'sha256')} "
        f"checkpoint_sha256={actual_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
