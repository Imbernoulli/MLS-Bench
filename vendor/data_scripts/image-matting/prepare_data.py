#!/usr/bin/env python3
"""Validate a pinned offline Adobe Composition-1K full-protocol staging tree.

The Adobe data is licensed and is therefore not downloaded by this script. Image
preparation must stage `composition1k/train/manifest.json` and
`composition1k/test/manifest.json` plus their referenced files before building the
per-repo image. It must also stage a source-authored `composition1k/manifest.json`
whose digest is pinned below before the package can ship. Final verification performs
no extraction, download, or conversion.

Each path field in a split manifest must have a sibling `<field>_sha256` value.
Paths are relative to the split directory and must stay inside that directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PROTOCOL = "composition1k-full-v1"
SOURCE_REVISION = "adobe-composition-1k-licensed-full-v1"
# Backfill this from the licensed, source-authored top-level manifest. An empty value
# deliberately blocks preparation; a digest generated from the staging tree itself is
# not an independent trusted reference.
EXPECTED_SOURCE_MANIFEST_SHA256 = ""
EXPECTED_COUNTS = {"train": 43_100, "test": 1_000}
REQUIRED_FIELDS = {
    "train": {"image", "alpha", "foreground", "background"},
    "test": {"image", "alpha"},
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise SystemExit(f"{label} must be one lowercase SHA-256 digest")
    return value


def _resolve_member(split_dir: Path, value: object, expected_sha256: object) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit("manifest path fields must be non-empty strings")
    candidate = (split_dir / value).resolve()
    try:
        candidate.relative_to(split_dir.resolve())
    except ValueError as exc:
        raise SystemExit(f"manifest path escapes split root: {value}") from exc
    if not candidate.is_file():
        raise SystemExit(f"manifest member is missing: {candidate}")
    expected = _required_sha256(expected_sha256, f"digest for {value}")
    observed = _sha256(candidate)
    if observed != expected:
        raise SystemExit(
            f"manifest member digest mismatch for {candidate}: "
            f"expected {expected}, got {observed}"
        )
    return candidate


def _validate_split(root: Path, split: str, proof: object) -> dict:
    split_dir = root / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path}")
    expected = EXPECTED_COUNTS[split]
    if not isinstance(proof, dict) or int(proof.get("count", -1)) != expected:
        raise SystemExit(f"source manifest does not bind {expected} {split} records")
    expected_manifest_sha256 = _required_sha256(
        proof.get("manifest_sha256"), f"{split} manifest digest"
    )
    observed_manifest_sha256 = _sha256(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise SystemExit(
            f"{split} manifest digest mismatch: expected {expected_manifest_sha256}, "
            f"got {observed_manifest_sha256}"
        )

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected:
        raise SystemExit(f"{split} requires exactly {expected} records")

    seen_image_paths = set()
    seen_image_digests = set()
    required = REQUIRED_FIELDS[split]
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise SystemExit(f"malformed {split} manifest row {index}")
        members = {}
        for field in sorted(required):
            members[field] = _resolve_member(
                split_dir, row[field], row.get(f"{field}_sha256")
            )
        image_path = str(members["image"].relative_to(split_dir.resolve()))
        image_digest = row["image_sha256"]
        if image_path in seen_image_paths or image_digest in seen_image_digests:
            raise SystemExit(f"duplicate {split} composite at row {index}")
        seen_image_paths.add(image_path)
        seen_image_digests.add(image_digest)

    return {
        "count": expected,
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": observed_manifest_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.data_root / "image-matting" / "composition1k"
    manifest_path = root / "manifest.json"
    expected_source_sha256 = _required_sha256(
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "EXPECTED_SOURCE_MANIFEST_SHA256 (operator backfill required)",
    )
    if not manifest_path.is_file():
        raise SystemExit(f"missing source-authored manifest: {manifest_path}")
    observed_source_sha256 = _sha256(manifest_path)
    if observed_source_sha256 != expected_source_sha256:
        raise SystemExit(
            "source manifest digest mismatch: "
            f"expected {expected_source_sha256}, got {observed_source_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("revision") != SOURCE_REVISION
        or int(manifest.get("crop_size", -1)) != 320
        or not isinstance(manifest.get("splits"), dict)
    ):
        raise SystemExit("unexpected Composition-1K source manifest protocol")
    splits = {
        split: _validate_split(root, split, manifest["splits"].get(split))
        for split in ("train", "test")
    }
    print(
        "MATTING_PREPARE_COMPLETE "
        f"protocol={PROTOCOL} train={splits['train']['count']} "
        f"test={splits['test']['count']} manifest_sha256={observed_source_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
