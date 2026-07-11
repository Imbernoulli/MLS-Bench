#!/usr/bin/env python3
"""Write canonical MT manifests into an OCI filesystem layer."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_common(path: Path):
    spec = importlib.util.spec_from_file_location("mt_layer_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MT common module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", type=Path, required=True)
    parser.add_argument("--layer-root", type=Path, required=True)
    args = parser.parse_args()

    common = _load_common(args.common)
    mt_root = args.layer_root / "data" / "machine-translation"
    for direction, (model_dir_name, _split_name) in common.DIRECTIONS.items():
        model_dir = mt_root / "models" / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_manifest.json").write_bytes(
            common._canonical_json_bytes(common.expected_model_manifest(direction))
        )
    data_dir = mt_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "source_manifest.json").write_bytes(
        common._canonical_json_bytes(common.expected_source_manifest())
    )
    print("MT_MANIFEST_LAYER_WRITTEN models=3 source_manifest=1", flush=True)


if __name__ == "__main__":
    main()
