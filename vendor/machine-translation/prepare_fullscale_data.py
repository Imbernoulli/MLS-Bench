#!/usr/bin/env python3
"""Prepare the complete pinned OPUS-100 test splits used by mt-* tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REPO_ID = "Helsinki-NLP/opus-100"
REVISION = "805090dc28bf78897da9641cdf08b61287580df9"
EXPECTED_ROWS = 2_000
SPLITS = {
    "de_en": {
        "path": "de-en/test-00000-of-00001.parquet",
        "source": "de",
        "target": "en",
        "parquet_sha256": "05913515e9dc8c11bc03570bd00ae5b551c32b03e07901369f91372ad63a3f11",
        "jsonl_sha256": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
    },
    "fr_en": {
        "path": "en-fr/test-00000-of-00001.parquet",
        "source": "fr",
        "target": "en",
        "parquet_sha256": "6e5862c14744efb89cf4c807cf0fd1a5969249935f21a1d03f3fbdbc0fb81971",
        "jsonl_sha256": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
    },
    "ru_en": {
        "path": "en-ru/test-00000-of-00001.parquet",
        "source": "ru",
        "target": "en",
        "parquet_sha256": "96bf7751ebd69615e1377a06cf49bb7d2d153124c77764620de435d0afc71935",
        "jsonl_sha256": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_parquet(spec: dict[str, str], source_root: Path | None) -> Path:
    relative = Path(spec["path"])
    if source_root is not None:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source parquet: {path}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=spec["path"],
            repo_type="dataset",
            revision=REVISION,
            local_files_only=False,
        )
    )


def prepare_split(name: str, spec: dict[str, str], source_root: Path | None,
                  output_root: Path) -> dict[str, object]:
    import pyarrow.parquet as parquet

    source_path = resolve_parquet(spec, source_root)
    actual_source_hash = sha256(source_path)
    if actual_source_hash != spec["parquet_sha256"]:
        raise ValueError(
            f"{name} parquet digest mismatch: expected {spec['parquet_sha256']}, "
            f"got {actual_source_hash}"
        )

    rows = parquet.read_table(source_path).to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"{name} row-count mismatch: expected {EXPECTED_ROWS}, got {len(rows)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{name}_test.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"translation"}:
                raise ValueError(f"{name} row {index} has an invalid top-level schema")
            translation = row["translation"]
            if not isinstance(translation, dict):
                raise ValueError(f"{name} row {index} has no translation mapping")
            source = translation.get(spec["source"])
            reference = translation.get(spec["target"])
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"{name} row {index} has an empty source")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"{name} row {index} has an empty reference")
            handle.write(
                json.dumps({"src": source, "ref": reference}, ensure_ascii=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())

    actual_output_hash = sha256(temporary_path)
    if actual_output_hash != spec["jsonl_sha256"]:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"{name} JSONL digest mismatch: expected {spec['jsonl_sha256']}, "
            f"got {actual_output_hash}"
        )
    temporary_path.replace(output_path)
    return {
        "direction": name,
        "rows": len(rows),
        "source_language": spec["source"],
        "target_language": spec["target"],
        "source_path": spec["path"],
        "source_sha256": actual_source_hash,
        "output_file": output_path.name,
        "output_sha256": actual_output_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Root containing the pinned repository paths; otherwise download via HF Hub",
    )
    args = parser.parse_args()

    records = [
        prepare_split(name, spec, args.source_root, args.output_root)
        for name, spec in SPLITS.items()
    ]
    manifest = {
        "schema_version": 1,
        "dataset": REPO_ID,
        "revision": REVISION,
        "expected_rows_per_direction": EXPECTED_ROWS,
        "splits": records,
    }
    manifest_path = args.output_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
