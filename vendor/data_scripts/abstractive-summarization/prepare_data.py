#!/usr/bin/env python3
"""Stage pinned models and complete official summarization test splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


DATASETS = {
    "xsum": {
        "repo": "EdinburghNLP/xsum",
        "revision": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
        "file": "data/test-00000-of-00001.parquet",
        "source_sha256": "224e9dbc6fed987759c1954603b43cb280b8d475d78893779130aa707d967ed7",
        "jsonl_sha256": "5b7854e6ee8b81493e38a5255ba794dec21735b30ba8f77280a394bf7d6dae53",
        "rows": 11334,
        "document": "document",
        "summary": "summary",
        "id": "id",
    },
    "cnndm": {
        "repo": "abisee/cnn_dailymail",
        "revision": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        "file": "3.0.0/test-00000-of-00001.parquet",
        "source_sha256": "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
        "jsonl_sha256": "525e5bf29cb49a1ea2c58939d11435b6d6422f9a59d9c15e43d226c464b33278",
        "rows": 11490,
        "document": "article",
        "summary": "highlights",
        "id": "id",
    },
    "samsum": {
        "repo": "knkarthick/samsum",
        "revision": "6b929ff10edec703164e3ddb2e94aae058c9ab5f",
        "file": "test.csv",
        "source_sha256": "0c1da58c77766d5eefaf89ee6c48133481df0730d706bcf826d01efd274d5e13",
        "jsonl_sha256": "b9dd8165689b1c252b2db617b089b408ea068ef5bd6819a93c64e8ebde856b86",
        "rows": 819,
        "document": "dialogue",
        "summary": "summary",
        "id": "id",
    },
}

MODELS = {
    "distilbart-xsum-12-6": {
        "repo": "sshleifer/distilbart-xsum-12-6",
        "revision": "5b2e376c845c201ddc34ec0e55fd1ad9890ba5ee",
        "weights_sha256": "6e9ebfc94e474225457570ad33c225cf66ca26279c5cd1cbfb67e089a03a791b",
        "weights_bytes": 611201041,
    },
    "distilbart-cnn-12-6": {
        "repo": "sshleifer/distilbart-cnn-12-6",
        "revision": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "weights_sha256": "3bac65d18c99463302d12ca75c2220ea714f9c81ce235f205fa818efe71df6ea",
        "weights_bytes": 1222317369,
    },
    "bart-large-cnn-samsum": {
        "repo": "philschmid/bart-large-cnn-samsum",
        "revision": "e49b3d60d923f12db22bdd363356f1a4c68532ad",
        "weights_sha256": "9f453aa6edef4dba1893723b7313b57b06b60214442d308a8acc3baa9583dd7b",
        "weights_bytes": 1625565295,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(models_dir: Path) -> dict[str, dict[str, object]]:
    from huggingface_hub import snapshot_download

    manifest: dict[str, dict[str, object]] = {}
    for name, spec in MODELS.items():
        destination = models_dir / name
        snapshot_download(
            repo_id=str(spec["repo"]),
            revision=str(spec["revision"]),
            local_dir=str(destination),
            ignore_patterns=[
                "*.onnx", "*.ot", "*.h5", "*.msgpack", "onnx/*",
                "openvino/*", "*.tflite", "flax_model.msgpack",
                "checkpoint-*", "checkpoint-*/*", ".git/*",
            ],
        )
        weights = destination / "pytorch_model.bin"
        actual_sha = sha256(weights)
        if (weights.stat().st_size != spec["weights_bytes"]
                or actual_sha != spec["weights_sha256"]):
            raise RuntimeError(f"model inventory mismatch for {name}")
        manifest[name] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "weights_bytes": weights.stat().st_size,
            "weights_sha256": actual_sha,
        }
        print(f"MODEL_READY name={name} sha256={actual_sha}", flush=True)
    return manifest


def _records(path: Path, spec: dict[str, object]):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)


def download_datasets(data_dir: Path, cache_dir: Path) -> dict[str, dict[str, object]]:
    from huggingface_hub import hf_hub_download

    manifest: dict[str, dict[str, object]] = {}
    for setting, spec in DATASETS.items():
        source = Path(hf_hub_download(
            repo_id=str(spec["repo"]),
            repo_type="dataset",
            filename=str(spec["file"]),
            revision=str(spec["revision"]),
            cache_dir=str(cache_dir),
        ))
        actual_source_sha = sha256(source)
        if actual_source_sha != spec["source_sha256"]:
            raise RuntimeError(f"source digest mismatch for {setting}")

        output = data_dir / f"{setting}_test.jsonl"
        row_count = 0
        with output.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(_records(source, spec)):
                converted = {
                    "id": str(record.get(str(spec["id"]), index)),
                    "document": str(record[str(spec["document"])]).strip(),
                    "summary": str(record[str(spec["summary"])]).strip(),
                }
                handle.write(json.dumps(
                    converted, ensure_ascii=False, separators=(",", ":")
                ) + "\n")
                row_count += 1
        actual_jsonl_sha = sha256(output)
        if row_count != spec["rows"] or actual_jsonl_sha != spec["jsonl_sha256"]:
            raise RuntimeError(f"converted inventory mismatch for {setting}")
        manifest[setting] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "source_file": spec["file"],
            "source_sha256": actual_source_sha,
            "rows": row_count,
            "jsonl_sha256": actual_jsonl_sha,
        }
        print(
            f"DATA_READY setting={setting} rows={row_count} "
            f"sha256={actual_jsonl_sha}",
            flush=True,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = Path(args.data_root) / "abstractive-summarization"
    models_dir = root / "models"
    data_dir = root / "data"
    cache_dir = root / "_download_cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "summ-full-official-test-v1",
        "models": download_models(models_dir),
        "datasets": download_datasets(data_dir, cache_dir),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ALL_DONE protocol=summ-full-official-test-v1", flush=True)


if __name__ == "__main__":
    main()
