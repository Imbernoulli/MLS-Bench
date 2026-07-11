#!/usr/bin/env python3
"""Build the pinned, full-scale NLI corpus used by every ``nli-*`` task.

The four source parquet files are immutable Hugging Face dataset artifacts.
Preparation preserves source order and text exactly, drops only SNLI rows whose
label is ``-1`` (no annotator consensus), and never samples or balances data.
The emitted manifest authenticates source files, canonical JSONL outputs, and the
five runtime model/tokenizer assets baked into the per-repository image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


PROTOCOL = "nli-full-snli-distilbert-v1"
MODEL_REPO = "distilbert/distilbert-base-uncased"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MODEL_FILES = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
SNLI_REPO = "stanfordnlp/snli"
SNLI_REVISION = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
MNLI_REPO = "nyu-mll/multi_nli"
MNLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"

# Hugging Face integer label ids for both corpora. SNLI's -1 means that no
# consensus label was assigned and is the only row type omitted from output.
ID2NAME = {0: "entailment", 1: "neutral", 2: "contradiction"}

SPLITS = {
    "snli_train": {
        "source": "snli_train.parquet",
        "source_hf_path": "plain_text/train-00000-of-00001.parquet",
        "source_repo": SNLI_REPO,
        "source_revision": SNLI_REVISION,
        "source_rows": 550152,
        "source_sha256": "ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620",
        "output": "snli/train.jsonl",
        "output_rows": 549367,
        "output_sha256": "3cdde4e94e0c5ca8e7e3d95b0c7c7b9fc03b101d3b9e79c422150bf5c17f1f73",
        "dropped_unlabeled": 785,
        "label_counts": {"entailment": 183416, "neutral": 182764, "contradiction": 183187},
    },
    "snli_test": {
        "source": "snli_test.parquet",
        "source_hf_path": "plain_text/test-00000-of-00001.parquet",
        "source_repo": SNLI_REPO,
        "source_revision": SNLI_REVISION,
        "source_rows": 10000,
        "source_sha256": "4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c",
        "output": "snli/test.jsonl",
        "output_rows": 9824,
        "output_sha256": "e30ea21eb677dab4806e1cc4c646dffc23985ffd982fd6bd15ab3617cd601dd8",
        "dropped_unlabeled": 176,
        "label_counts": {"entailment": 3368, "neutral": 3219, "contradiction": 3237},
    },
    "mnli_matched": {
        "source": "mnli_matched.parquet",
        "source_hf_path": "data/validation_matched-00000-of-00001.parquet",
        "source_repo": MNLI_REPO,
        "source_revision": MNLI_REVISION,
        "source_rows": 9815,
        "source_sha256": "350c26950b55f460b50d36c76aef87d64b49c78812d7abf7bf97e5fede10f186",
        "output": "mnli/dev_matched.jsonl",
        "output_rows": 9815,
        "output_sha256": "a612ccdf07b2fbe73e2904b061b9e278f552a39b553999bc626de6df6ec4b66d",
        "dropped_unlabeled": 0,
        "label_counts": {"entailment": 3479, "neutral": 3123, "contradiction": 3213},
    },
    "mnli_mismatched": {
        "source": "mnli_mismatched.parquet",
        "source_hf_path": "data/validation_mismatched-00000-of-00001.parquet",
        "source_repo": MNLI_REPO,
        "source_revision": MNLI_REVISION,
        "source_rows": 9832,
        "source_sha256": "6b0e1231cebedd255000a7d1732af37ad0500902239db625e231ed77c0c8f2f8",
        "output": "mnli/dev_mismatched.jsonl",
        "output_rows": 9832,
        "output_sha256": "a08757b4ddc34421f8f6eac69eb5dd97b2125693078c541cad2d54689013f68d",
        "dropped_unlabeled": 0,
        "label_counts": {"entailment": 3463, "neutral": 3129, "contradiction": 3240},
    },
}

# This deterministic diagnostic is retained for the hypothesis-bias sibling,
# which scores the complete splits by default. It is not used to select or drop
# any row and is not part of the representative finetuning score.
_NEG_CUES = (
    "no ", "not ", "n't", "nobody", "never", "nothing", "none", "empty",
    "sleeping", "cannot",
)
_NEUTRAL_CUES = (
    "tall", "sad", "happy", "first", "because", "some ", "many", "vacation",
    "competition", "won ", "birthday", "favorite",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hypothesis_only_prediction(hypothesis: str) -> str:
    text = " " + hypothesis.lower().strip() + " "
    if any(cue in text for cue in _NEG_CUES):
        return "contradiction"
    if any(cue in text for cue in _NEUTRAL_CUES):
        return "neutral"
    return "entailment"


def _build_split(source: Path, destination: Path, expected: dict) -> dict:
    import pyarrow.parquet as parquet

    actual_source_sha256 = _sha256(source)
    if actual_source_sha256 != expected["source_sha256"]:
        raise SystemExit(
            f"source digest mismatch for {source.name}: {actual_source_sha256} != "
            f"{expected['source_sha256']}"
        )

    table = parquet.read_table(source, columns=["premise", "hypothesis", "label"])
    if table.num_rows != expected["source_rows"]:
        raise SystemExit(
            f"source row mismatch for {source.name}: {table.num_rows} != "
            f"{expected['source_rows']}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    hard_count = 0
    dropped = 0
    output_rows = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for batch in table.to_batches(max_chunksize=8192):
            for record in batch.to_pylist():
                label_id = int(record["label"])
                if label_id == -1:
                    dropped += 1
                    continue
                if label_id not in ID2NAME:
                    raise SystemExit(f"unsupported label {label_id} in {source.name}")
                premise = record["premise"]
                hypothesis = record["hypothesis"]
                if not isinstance(premise, str) or not premise.strip():
                    raise SystemExit(f"empty premise in {source.name} row {output_rows + dropped}")
                if not isinstance(hypothesis, str) or not hypothesis.strip():
                    raise SystemExit(f"empty hypothesis in {source.name} row {output_rows + dropped}")
                label = ID2NAME[label_id]
                hard = _hypothesis_only_prediction(hypothesis) != label
                row = {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                    "hard": hard,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts[label] += 1
                hard_count += int(hard)
                output_rows += 1

    expected_counts = expected["label_counts"]
    if output_rows != expected["output_rows"]:
        raise SystemExit(
            f"output row mismatch for {source.name}: {output_rows} != "
            f"{expected['output_rows']}"
        )
    if dropped != expected["dropped_unlabeled"]:
        raise SystemExit(
            f"unlabeled row mismatch for {source.name}: {dropped} != "
            f"{expected['dropped_unlabeled']}"
        )
    if dict(counts) != expected_counts:
        raise SystemExit(
            f"label inventory mismatch for {source.name}: {dict(counts)} != "
            f"{expected_counts}"
        )

    output_sha256 = _sha256(destination)
    if output_sha256 != expected["output_sha256"]:
        raise SystemExit(
            f"canonical output digest mismatch for {source.name}: "
            f"{output_sha256} != {expected['output_sha256']}"
        )

    return {
        "source_file": expected["source"],
        "source_repo": expected["source_repo"],
        "source_revision": expected["source_revision"],
        "source_rows": table.num_rows,
        "source_sha256": actual_source_sha256,
        "output_file": expected["output"],
        "output_rows": output_rows,
        "output_sha256": output_sha256,
        "dropped_unlabeled": dropped,
        "label_counts": dict(counts),
        "diagnostic_hard_rows": hard_count,
    }


def _download_pinned_source(expected: dict, cache_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    endpoint = os.environ.get("HF_ENDPOINT")
    path = hf_hub_download(
        repo_id=expected["source_repo"],
        filename=expected["source_hf_path"],
        revision=expected["source_revision"],
        repo_type="dataset",
        cache_dir=cache_root,
        endpoint=endpoint,
    )
    return Path(path)


def _download_pinned_model_file(filename: str, cache_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        revision=MODEL_REVISION,
        cache_dir=cache_root,
        endpoint=os.environ.get("HF_ENDPOINT"),
    )
    return Path(path)


def _stage_model_assets(
    destination: Path,
    cache_root: Path,
    source_root: Path | None = None,
) -> dict[str, str]:
    """Stage the exact offline runtime assets during one-time preparation."""
    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for filename, expected_digest in MODEL_FILES.items():
        target = destination / filename
        if target.is_file() and _sha256(target) == expected_digest:
            staged[filename] = expected_digest
            print(
                f"NLI_MODEL_ASSET file={filename} sha256={expected_digest}",
                flush=True,
            )
            continue
        source = (
            source_root / filename
            if source_root is not None
            else _download_pinned_model_file(filename, cache_root)
        )
        if not source.is_file():
            raise SystemExit(f"required model source is missing: {source}")
        source_digest = _sha256(source)
        if source_digest != expected_digest:
            raise SystemExit(
                f"model asset digest mismatch for {filename}: "
                f"{source_digest} != {expected_digest}"
            )
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        target_digest = _sha256(target)
        if target_digest != expected_digest:
            raise SystemExit(
                f"staged model asset digest mismatch for {filename}: "
                f"{target_digest} != {expected_digest}"
            )
        staged[filename] = target_digest
        print(
            f"NLI_MODEL_ASSET file={filename} sha256={target_digest}", flush=True
        )
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        help=("optional directory containing the four pinned parquet source files; "
              "when omitted, exact-revision artifacts are downloaded on the host"),
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="root under which natural-language-inference/data is written",
    )
    parser.add_argument(
        "--model-source-root", type=Path,
        help=("optional directory containing the five pinned runtime model and "
              "tokenizer files; otherwise exact-revision artifacts are downloaded"),
    )
    args = parser.parse_args()

    output_root = args.data_root / "natural-language-inference"
    data_root = output_root / "data"
    staged_model_files = _stage_model_assets(
        output_root / "models" / MODEL_NAME,
        output_root / ".source-cache" / "models",
        args.model_source_root,
    )
    manifest = {
        "protocol": PROTOCOL,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "files_sha256": staged_model_files,
        },
        "splits": {},
    }
    for name, expected in SPLITS.items():
        if args.source_root is not None:
            source = args.source_root / expected["source"]
        else:
            source = _download_pinned_source(
                expected, output_root / ".source-cache"
            )
        result = _build_split(
            source,
            data_root / expected["output"],
            expected,
        )
        manifest["splits"][name] = result
        print(
            f"NLI_BUILT split={name} rows={result['output_rows']} "
            f"sha256={result['output_sha256']}",
            flush=True,
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"NLI_MANIFEST sha256={_sha256(manifest_path)}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
