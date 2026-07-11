#!/usr/bin/env python3
"""Build the pinned full-resolution Places365 inpainting data artifact.

This is an image-build/data-staging program. It is never invoked by a task's
verification command. The artifact uses every image in the canonical Places365
``val_256`` archive without resizing: 32,000 images for optimization and 4,500
disjoint images for evaluation. The split and all provenance fields are fixed.

This protocol is a research-scale inpainting proxy over a canonical real-image
archive. It is not claimed to reproduce the much larger Places2 training corpus
used by DeepFillv2.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

PROTOCOL_ID = "places365-val256-fullres-v1"
SCHEMA_VERSION = 1
IMAGE_SIZE = 256
TRAIN_COUNT = 32_000
VAL_COUNT = 4_500
ARCHIVE_IMAGE_COUNT = TRAIN_COUNT + VAL_COUNT
SPLIT_SEED = 20_260_705
MASK_SEED = 999
TRAIN_STEPS = 100_000
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

PLACES_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
PLACES_TAR_NAME = "val_256.tar"
PLACES_TAR_BYTES = 525_158_400
PLACES_TAR_MD5 = "e27b17d8d44f4af9a78502beb927f808"
PLACES_TAR_SHA256 = "24b4e639ef12a0012af525bc4cb443e4ab4aaea8369a1fb009b70e4a4aad5d48"

PROTOCOL_MANIFEST = "protocol_manifest.json"


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _file_digests(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - compare with the dataset's published digest
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _validated_members(tar_path: Path) -> list[tarfile.TarInfo]:
    if not tar_path.is_file() or tar_path.stat().st_size != PLACES_TAR_BYTES:
        raise RuntimeError(
            f"invalid {PLACES_TAR_NAME} size; expected exactly {PLACES_TAR_BYTES} bytes"
        )
    actual_md5, actual_sha256 = _file_digests(tar_path)
    if actual_md5 != PLACES_TAR_MD5 or actual_sha256 != PLACES_TAR_SHA256:
        raise RuntimeError(
            f"{PLACES_TAR_NAME} provenance mismatch: md5={actual_md5} "
            f"sha256={actual_sha256}"
        )
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.lower().endswith(".jpg")
                ),
                key=lambda member: member.name,
            )
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"cannot read canonical Places365 archive: {exc}") from exc
    names = [member.name for member in members]
    if len(names) != ARCHIVE_IMAGE_COUNT or len(set(names)) != ARCHIVE_IMAGE_COUNT:
        raise RuntimeError(
            f"archive has {len(names)} unique regular JPEG entries; "
            f"expected {ARCHIVE_IMAGE_COUNT}"
        )
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError(f"unsafe archive member: {name!r}")
    return members


def _ensure_archive(raw_dir: Path) -> tuple[Path, list[tarfile.TarInfo]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_path = raw_dir / PLACES_TAR_NAME
    if final_path.exists():
        return final_path, _validated_members(final_path)

    partial_path = raw_dir / f"{PLACES_TAR_NAME}.part"
    subprocess.run(
        [
            "curl",
            "-fSL",
            "--retry",
            "12",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--max-time",
            "3600",
            "-o",
            str(partial_path),
            PLACES_URL,
        ],
        check=True,
    )
    members = _validated_members(partial_path)
    partial_path.replace(final_path)
    return final_path, members


def _validate_jpeg(raw: bytes, source: str) -> None:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(
                    f"{source!r} has size {image.size}; expected {(IMAGE_SIZE, IMAGE_SIZE)}"
                )
            if image.convert("RGB").size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"cannot decode {source!r} as RGB")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid JPEG {source!r}: {exc}") from exc


def _extract_split(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    split_name: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    image_dir = output_dir / split_name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, member in enumerate(members):
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"archive member is unreadable: {member.name!r}")
        raw = extracted.read()
        _validate_jpeg(raw, member.name)
        relpath = f"{split_name}/images/{index:05d}.jpg"
        (output_dir / relpath).write_bytes(raw)
        manifest.append(
            {
                "index": index,
                "path": relpath,
                "source": member.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (output_dir / split_name / "manifest.json").write_bytes(
        _canonical_json_bytes(manifest)
    )
    return manifest


def _expected_protocol_manifest(
    train_manifest_sha256: str,
    val_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_ID,
        "dataset": "Places365-Standard",
        "source_split": "val_256",
        "source_url": PLACES_URL,
        "archive_bytes": PLACES_TAR_BYTES,
        "archive_md5": PLACES_TAR_MD5,
        "archive_sha256": PLACES_TAR_SHA256,
        "archive_image_count": ARCHIVE_IMAGE_COUNT,
        "split_seed": SPLIT_SEED,
        "mask_seed": MASK_SEED,
        "image_size": IMAGE_SIZE,
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "train_steps": TRAIN_STEPS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "train_manifest_sha256": train_manifest_sha256,
        "val_manifest_sha256": val_manifest_sha256,
    }


def _validate_split(root: Path, split_name: str, expected_count: int) -> str:
    manifest_path = root / split_name / "manifest.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, list) or len(manifest) != expected_count:
        raise RuntimeError(f"invalid {split_name} manifest count")
    seen_sources: set[str] = set()
    for expected_index, item in enumerate(manifest):
        if not isinstance(item, dict) or set(item) != {
            "index", "path", "source", "bytes", "sha256"
        }:
            raise RuntimeError(f"malformed {split_name} manifest row {expected_index}")
        if item["index"] != expected_index or item["source"] in seen_sources:
            raise RuntimeError(f"duplicate or out-of-order {split_name} row {expected_index}")
        seen_sources.add(item["source"])
        relpath = PurePosixPath(item["path"])
        if (
            relpath.is_absolute()
            or ".." in relpath.parts
            or len(relpath.parts) != 3
            or relpath.parts[:2] != (split_name, "images")
        ):
            raise RuntimeError(f"unsafe prepared path in row {expected_index}")
        path = root.joinpath(*relpath.parts)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing or truncated artifact {item['path']!r}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"artifact digest mismatch for {item['path']!r}")
    if raw_manifest != _canonical_json_bytes(manifest):
        raise RuntimeError(f"{split_name} manifest is not canonical JSON")
    return hashlib.sha256(raw_manifest).hexdigest()


def _prepared_is_valid(root: Path) -> bool:
    try:
        train_sha = _validate_split(root, "train", TRAIN_COUNT)
        val_sha = _validate_split(root, "val", VAL_COUNT)
        train = json.loads((root / "train" / "manifest.json").read_text())
        val = json.loads((root / "val" / "manifest.json").read_text())
        if {row["source"] for row in train} & {row["source"] for row in val}:
            return False
        expected = _expected_protocol_manifest(train_sha, val_sha)
        raw_protocol = (root / PROTOCOL_MANIFEST).read_bytes()
        return raw_protocol == _canonical_json_bytes(expected)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory containing a previously downloaded canonical val_256.tar",
    )
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve() / "image-inpainting" / "community_v1"
    if _prepared_is_valid(root):
        print(f"validated image-inpainting artifact at {root}")
        return

    raw_dir = args.raw_dir or (root.parent / "_raw")
    archive_path, members = _ensure_archive(raw_dir)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(members))
    train_members = [members[int(index)] for index in order[:TRAIN_COUNT]]
    val_members = [members[int(index)] for index in order[TRAIN_COUNT:]]

    staging = root.with_name(f"{root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:") as archive:
        train = _extract_split(archive, train_members, "train", staging)
        val = _extract_split(archive, val_members, "val", staging)
    if {row["source"] for row in train} & {row["source"] for row in val}:
        raise RuntimeError("train/validation source overlap")

    train_raw = (staging / "train" / "manifest.json").read_bytes()
    val_raw = (staging / "val" / "manifest.json").read_bytes()
    protocol = _expected_protocol_manifest(
        hashlib.sha256(train_raw).hexdigest(),
        hashlib.sha256(val_raw).hexdigest(),
    )
    (staging / PROTOCOL_MANIFEST).write_bytes(_canonical_json_bytes(protocol))
    if not _prepared_is_valid(staging):
        raise RuntimeError("post-build image-inpainting integrity check failed")

    if root.exists():
        shutil.rmtree(root)
    staging.replace(root)
    print(
        f"image-inpainting artifact ready: protocol={PROTOCOL_ID} "
        f"train={TRAIN_COUNT} val={VAL_COUNT} resolution={IMAGE_SIZE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
