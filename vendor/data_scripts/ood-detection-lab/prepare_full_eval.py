#!/usr/bin/env python3
"""Build the full-image CIFAR-10 OOD evaluation inventory.

This is a staging-time command.  It reads canonical dataset releases already
present on a network-enabled development machine and writes one offline uint8
archive.  Final verification performs no download, installation, extraction,
or compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path

import numpy as np


PROTOCOL = "openood_cifar10_resnet18_full_v1"
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
EXPECTED_SOURCE_SHA256 = {
    "svhn": "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b",
    "cifar100": "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68",
    "tiny_imagenet": "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} digest mismatch: got {observed}, expected {expected}")


def load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    from PIL import Image

    manifest = hashlib.sha256()

    def load_split(split: str, expected_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for label, class_name in enumerate(CLASSES):
            directory = root / split / class_name
            paths = sorted(directory.glob("*.png"))
            if len(paths) != expected_per_class:
                raise RuntimeError(
                    f"CIFAR-10 {split}/{class_name} has {len(paths)} images, "
                    f"expected {expected_per_class}"
                )
            for path in paths:
                relative = path.relative_to(root).as_posix().encode("utf-8")
                encoded = path.read_bytes()
                manifest.update(len(relative).to_bytes(4, "big"))
                manifest.update(relative)
                manifest.update(len(encoded).to_bytes(8, "big"))
                manifest.update(encoded)
                image = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"), dtype=np.uint8)
                if image.shape != (32, 32, 3):
                    raise RuntimeError(f"CIFAR-10 image has shape {image.shape}: {path}")
                images.append(image.transpose(2, 0, 1))
                labels.append(label)
        return np.stack(images), np.asarray(labels, dtype=np.int64)

    train_images, train_labels = load_split("train", 5_000)
    test_images, test_labels = load_split("test", 1_000)
    return train_images, train_labels, test_images, test_labels, manifest.hexdigest()


def load_svhn(path: Path) -> np.ndarray:
    import scipy.io as sio

    value = sio.loadmat(path)["X"]
    if value.shape != (32, 32, 3, 26_032) or value.dtype != np.uint8:
        raise RuntimeError(f"SVHN test inventory has unexpected shape/dtype: {value.shape} {value.dtype}")
    return np.transpose(value, (3, 2, 0, 1)).copy()


def load_cifar100(path: Path) -> np.ndarray:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path, columns=["img"])
    if table.num_rows != 10_000:
        raise RuntimeError(f"CIFAR-100 test inventory has {table.num_rows} rows, expected 10000")
    output = []
    for item in table.column("img").to_pylist():
        encoded = item["bytes"] if isinstance(item, dict) else item
        image = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"), dtype=np.uint8)
        if image.shape != (32, 32, 3):
            raise RuntimeError(f"CIFAR-100 image has unexpected shape {image.shape}")
        output.append(image.transpose(2, 0, 1))
    return np.stack(output)


def load_tiny_imagenet(val_root: Path) -> np.ndarray:
    from PIL import Image

    paths = sorted((val_root / "images").glob("*.JPEG"))
    if len(paths) != 10_000:
        raise RuntimeError(f"Tiny-ImageNet val has {len(paths)} images, expected 10000")
    output = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        value = np.asarray(image, dtype=np.uint8)
        if value.shape != (32, 32, 3):
            raise RuntimeError(f"Tiny-ImageNet image has unexpected shape {value.shape}: {path}")
        output.append(value.transpose(2, 0, 1))
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cifar10-root", type=Path,
        default=Path("/home/lvbohan/onboarding/laplace-b0/cifar10"),
    )
    parser.add_argument(
        "--svhn", type=Path,
        default=Path("/home/lvbohan/onboarding/ood-onboarding/raw/svhn_test_32x32.mat"),
    )
    parser.add_argument(
        "--cifar100", type=Path,
        default=Path("/home/lvbohan/onboarding/ood-onboarding/raw/cifar100_test.parquet"),
    )
    parser.add_argument(
        "--tiny-imagenet-zip", type=Path,
        default=Path("/home/lvbohan/onboarding/ood-onboarding/raw/tiny-imagenet-200.zip"),
    )
    parser.add_argument(
        "--tiny-imagenet-val", type=Path,
        default=Path(
            "/home/lvbohan/onboarding/ood-onboarding/raw/"
            "tiny-imagenet-200_extracted/tiny-imagenet-200/val"
        ),
    )
    args = parser.parse_args()

    require_digest(args.svhn, EXPECTED_SOURCE_SHA256["svhn"], "SVHN source")
    require_digest(args.cifar100, EXPECTED_SOURCE_SHA256["cifar100"], "CIFAR-100 source")
    require_digest(
        args.tiny_imagenet_zip,
        EXPECTED_SOURCE_SHA256["tiny_imagenet"],
        "Tiny-ImageNet source",
    )

    train_images, train_labels, id_images, id_labels, cifar10_manifest = load_cifar10(
        args.cifar10_root
    )
    arrays = {
        "train_images": train_images,
        "train_labels": train_labels,
        "id_images": id_images,
        "id_labels": id_labels,
        "ood_svhn_images": load_svhn(args.svhn),
        "ood_cifar100_images": load_cifar100(args.cifar100),
        "ood_tin_images": load_tiny_imagenet(args.tiny_imagenet_val),
    }
    expected = {
        "train_images": (50_000, 3, 32, 32),
        "train_labels": (50_000,),
        "id_images": (10_000, 3, 32, 32),
        "id_labels": (10_000,),
        "ood_svhn_images": (26_032, 3, 32, 32),
        "ood_cifar100_images": (10_000, 3, 32, 32),
        "ood_tin_images": (10_000, 3, 32, 32),
    }
    for name, shape in expected.items():
        value = arrays[name]
        if value.shape != shape:
            raise RuntimeError(f"{name} has shape {value.shape}, expected {shape}")
        required_dtype = np.int64 if name.endswith("labels") else np.uint8
        if value.dtype != required_dtype:
            raise RuntimeError(f"{name} has dtype {value.dtype}, expected {required_dtype}")
    for name in ("train_labels", "id_labels"):
        labels = arrays[name]
        if labels.min() != 0 or labels.max() != 9 or not np.array_equal(
            np.bincount(labels, minlength=10),
            np.full(10, labels.size // 10, dtype=np.int64),
        ):
            raise RuntimeError(f"{name} does not have the canonical balanced CIFAR-10 labels")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray(1, dtype=np.int64),
            protocol=np.asarray(PROTOCOL),
            cifar10_png_manifest_sha256=np.asarray(cifar10_manifest),
            svhn_source_sha256=np.asarray(EXPECTED_SOURCE_SHA256["svhn"]),
            cifar100_source_sha256=np.asarray(EXPECTED_SOURCE_SHA256["cifar100"]),
            tiny_imagenet_source_sha256=np.asarray(EXPECTED_SOURCE_SHA256["tiny_imagenet"]),
            **arrays,
        )
    os.replace(temporary, args.output)
    print(
        f"OOD_FULL_DATA_READY protocol={PROTOCOL} path={args.output} "
        f"sha256={sha256(args.output)} n_train=50000 n_id=10000 "
        "n_svhn=26032 n_cifar100=10000 n_tin=10000",
        flush=True,
    )


if __name__ == "__main__":
    main()
