#!/usr/bin/env python3
"""Prepare the three frozen models required by text-simplification tasks.

Every repo/revision and canonical runtime file digest is fixed below. The capacity
task needs both T5-small checkpoints and the T5-base checkpoint; all other tasks
use the base Turk checkpoint.

## Held-out-reference hygiene (class-3 fix, 2026-07-05, PR #54 pattern)

This script is a pkg_config ``data_deps[].prepare`` entry, i.e. its OUTPUT
directory (``{data_root}/text-simplification``) is exactly what
``harbor_adapter/scripts/build_base_image.py`` bakes into the SHARED base
image every simp-* task's agent shell can read. It therefore now stages the
MODEL ONLY -- no test-set JSONL of any kind. The three FIXED simplification
test sets (source sentences + held-out human reference simplifications, from
GEM/wiki_auto_asset_turk) are built separately by
``holdout/text-simplification/generate_data.py``, which writes the
sources-only half to the agent-visible, git-checked-in
``vendor/text-simplification/_simp_data/`` and the references-only half
(the literal SARI answer key) DIRECTLY into each shipped ``tasks/simp-*/
data/`` dir (verifier-only, mounted at ``$TASK_DIR/data`` only at scoring
time). See ``vendor/text-simplification/common.py::load_dataset`` /
``_refs_data_path`` for the loader side of this split.

Requires network on the HOST; the task container is offline.
"""
import argparse
import hashlib
from pathlib import Path

MODELS = {
    "t5-small-finetuned-turk-text-simplification": {
        "repo_id": "mrm8488/t5-small-finetuned-turk-text-simplification",
        "revision": "f1c6a63751592c9b51d27acce8ab77e02563c983",
        "files": {
            "config.json": "0e29a6b11425fd91dc3f3e80f55aa6e38f499e08eedb5812d002666f418fe10e",
            "tokenizer.json": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            "model.safetensors": "4d4e5fcb2ce1bb58134fb86179b01407dc6bc49370960f2f40fa006effba2d24",
        },
    },
    "t5-small-finetuned-text-simplification": {
        "repo_id": "mrm8488/t5-small-finetuned-text-simplification",
        "revision": "6b7f868dad51927dbf8fffd05bc8d71abe379c87",
        "files": {
            "config.json": "b08329c26cb26547cf83a44b03ad1d4407f8bf35d326952450de9d01427bcd90",
            "tokenizer.json": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            "pytorch_model.bin": "da7890699bcb91de81d46cabe175bbacff9d31d2e9badc83dc2ada3e2345ca88",
        },
    },
    "t5-base-finetuned-turk-text-simplification": {
        "repo_id": "mrm8488/t5-base-finetuned-turk-text-simplification",
        "revision": "3049a645d59a3bb39abfb808b2ac89896876980f",
        "files": {
            "config.json": "6603801d50185a7f2b8955fd3794b1e94813952b21fb1d25dfb6f6456231f20f",
            "tokenizer.json": "f9eadbc572afadad1a76d19519db98e9c03261adb016ada4568c28ef7dd3a854",
            "model.safetensors": "063b490a880f6164b5aa7e8bc470911825c981ebf7396a31a4c9bf9f642c7fb4",
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(dst: Path, files: dict[str, str]) -> bool:
    return all(
        (dst / filename).is_file()
        and _sha256(dst / filename) == expected_sha
        for filename, expected_sha in files.items()
    )


def download_model(models_dir: Path, directory: str, spec: dict) -> None:
    from huggingface_hub import snapshot_download

    dst = models_dir / directory
    if dst.exists() and _verify(dst, spec["files"]):
        print(f"model {spec['repo_id']} already verified", flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        local_dir=str(dst),
        allow_patterns=[
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "spiece.model", *spec["files"],
        ],
    )
    if not _verify(dst, spec["files"]):
        raise ValueError(f"downloaded model does not match pinned manifest: {dst}")
    print(f"downloaded and verified {spec['repo_id']}@{spec['revision']} -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root) / "text-simplification"
    (root / "models").mkdir(parents=True, exist_ok=True)
    for directory, spec in MODELS.items():
        download_model(root / "models", directory, spec)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
