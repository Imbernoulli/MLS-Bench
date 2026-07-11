#!/usr/bin/env python3
"""Build the immutable data/model layer for machine-translation tasks.

This script is for worker/image preparation, where network access is allowed.
Verification itself is offline and never invokes this script.  The builder pins
the OPUS-100 dataset and all three MarianMT repositories by commit, verifies the
source parquet and final JSONL digests, downloads only the exact runtime model
files, and writes the canonical manifests consumed by trusted ``common.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path


def _load_common():
    path = Path(__file__).resolve().parents[2] / "machine-translation" / "common.py"
    spec = importlib.util.spec_from_file_location("mlsbench_mt_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted machine-translation metadata: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()


def _verify_model_directory(direction: str, path: Path) -> None:
    spec = common.MODEL_SPECS[direction]
    expected_names = set(spec["files"]) | {"model_manifest.json"}
    actual_names = {item.name for item in path.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"unexpected model inventory for {direction}: expected "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )
    expected_manifest = common._canonical_json_bytes(
        common.expected_model_manifest(direction)
    )
    if (path / "model_manifest.json").read_bytes() != expected_manifest:
        raise ValueError(f"model manifest mismatch for {direction}")
    for name, (expected_size, expected_digest) in spec["files"].items():
        artifact = path / name
        if not artifact.is_file() or artifact.is_symlink():
            raise FileNotFoundError(f"missing regular model artifact: {artifact}")
        if artifact.stat().st_size != expected_size:
            raise ValueError(f"model artifact size mismatch: {artifact}")
        if common._sha256_file(artifact) != expected_digest:
            raise ValueError(f"model artifact digest mismatch: {artifact}")


def prepare_model(direction: str, models_root: Path) -> None:
    from huggingface_hub import snapshot_download

    model_dir, _output_file = common.DIRECTIONS[direction]
    destination = models_root / model_dir
    if destination.exists():
        _verify_model_directory(direction, destination)
        print(f"MODEL_VERIFIED direction={direction} path={destination}", flush=True)
        return

    spec = common.MODEL_SPECS[direction]
    temporary = models_root / f".{model_dir}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        snapshot_download(
            repo_id=spec["repository"],
            revision=spec["revision"],
            local_dir=str(temporary),
            allow_patterns=sorted(spec["files"]),
        )
        cache_dir = temporary / ".cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        actual_names = {item.name for item in temporary.iterdir()}
        if actual_names != set(spec["files"]):
            raise ValueError(
                f"downloaded model inventory mismatch for {direction}: "
                f"{sorted(actual_names)}"
            )
        for name, (expected_size, expected_digest) in spec["files"].items():
            artifact = temporary / name
            if not artifact.is_file() or artifact.is_symlink():
                raise FileNotFoundError(f"missing downloaded artifact: {artifact}")
            if artifact.stat().st_size != expected_size:
                raise ValueError(f"downloaded artifact size mismatch: {artifact}")
            if common._sha256_file(artifact) != expected_digest:
                raise ValueError(f"downloaded artifact digest mismatch: {artifact}")
        (temporary / "model_manifest.json").write_bytes(
            common._canonical_json_bytes(common.expected_model_manifest(direction))
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _verify_model_directory(direction, destination)
    print(f"MODEL_BUILT direction={direction} path={destination}", flush=True)


def _resolve_source(spec: dict, source_root: Path | None) -> Path:
    if source_root is not None:
        path = source_root / spec["source_path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source parquet: {path}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=common.DATASET_ID,
            filename=spec["source_path"],
            repo_type="dataset",
            revision=common.DATASET_REVISION,
            local_files_only=False,
        )
    )


def prepare_split(direction: str, data_root: Path,
                  source_root: Path | None) -> None:
    import pyarrow.parquet as parquet

    spec = common.DATA_SPECS[direction]
    source_path = _resolve_source(spec, source_root)
    source_digest = common._sha256_file(source_path)
    if source_digest != spec["source_sha256"]:
        raise ValueError(
            f"{direction} source digest mismatch: expected "
            f"{spec['source_sha256']}, got {source_digest}"
        )
    rows = parquet.read_table(source_path).to_pylist()
    if len(rows) != common.OFFICIAL_TEST_PAIRS:
        raise ValueError(
            f"{direction} row-count mismatch: expected "
            f"{common.OFFICIAL_TEST_PAIRS}, got {len(rows)}"
        )

    _model_dir, output_name = common.DIRECTIONS[direction]
    output_path = data_root / output_name
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"translation"}:
                raise ValueError(f"{direction} row {index} has an invalid schema")
            translation = row["translation"]
            if not isinstance(translation, dict):
                raise ValueError(f"{direction} row {index} has no translation mapping")
            source = translation.get(spec["source_language"])
            reference = translation.get(common.TGT_LANG)
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{direction} row {index} has an empty source")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{direction} row {index} has an empty reference")
            handle.write(
                json.dumps({"src": source, "ref": reference}, ensure_ascii=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    output_digest = common._sha256_file(temporary)
    if output_digest != spec["output_sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{direction} JSONL digest mismatch: expected "
            f"{spec['output_sha256']}, got {output_digest}"
        )
    temporary.replace(output_path)
    print(
        f"SPLIT_BUILT direction={direction} rows={len(rows)} "
        f"sha256={output_digest}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root containing pinned parquet paths; otherwise use HF Hub",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = args.data_root / "machine-translation"
    models_root = root / "models"
    data_root = root / "data"
    models_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    for direction in common.DIRECTIONS:
        prepare_model(direction, models_root)
        prepare_split(direction, data_root, args.source_root)

    (data_root / "source_manifest.json").write_bytes(
        common._canonical_json_bytes(common.expected_source_manifest())
    )
    expected_names = {
        output_file for _model_dir, output_file in common.DIRECTIONS.values()
    } | {"source_manifest.json"}
    actual_names = {item.name for item in data_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            f"unexpected final OPUS data inventory: expected "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )
    print(
        f"MT_ASSETS_COMPLETE source_manifest_sha256="
        f"{common.source_manifest_sha256()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
