#!/usr/bin/env python3
"""Stage the full MBPP-sanitized test protocol and frozen code model.

Image preparation may use the network; final verification may not. The scored
inventory is every item in the canonical sanitized test split (257 problems), in
source order. Each record exposes the first gold assertion to the selection policy
and reserves the remaining assertions for functional-correctness scoring.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


EXPECTED_PROBLEMS = 257
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "357b899b4714bf46d935fb9911e8139b5b9efc29"
DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
PROTOCOL = "mbpp-sanitized-reserved-v2"
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_point(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def _assert_calls_entry(assert_src: str, entry: str) -> bool:
    return re.search(rf"\b{re.escape(entry)}\s*\(", assert_src) is not None


def _assertion_fingerprint(assert_src: str) -> str:
    try:
        return ast.dump(ast.parse(assert_src), include_attributes=False)
    except SyntaxError as exc:
        raise SystemExit(f"invalid MBPP assertion: {assert_src!r}") from exc


def _stage_model(models_dir: Path) -> Path:
    dst = models_dir / "Qwen2.5-Coder-1.5B-Instruct"
    from huggingface_hub import snapshot_download

    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(dst),
        ignore_patterns=["*.onnx", "*.ot", "*.h5", "*.msgpack", "*.gguf", "onnx/*"],
    )
    missing = [name for name in MODEL_FILES if not (dst / name).is_file()]
    if missing:
        raise SystemExit(f"pinned model snapshot is incomplete: {missing}")
    model_manifest = {
        "files": {
            name: {
                "bytes": (dst / name).stat().st_size,
                "sha256": _sha256(dst / name),
            }
            for name in MODEL_FILES
        },
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    (dst / "model_manifest.json").write_text(_canonical_json(model_manifest))
    return dst


def _build_problems(data_dir: Path) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        "sanitized",
        split="test",
        revision=DATASET_REVISION,
    )
    if len(dataset) != EXPECTED_PROBLEMS:
        raise SystemExit(f"expected {EXPECTED_PROBLEMS} sanitized MBPP test rows, got {len(dataset)}")

    problems = []
    for source_index, row in enumerate(dataset):
        code = str(row.get("code", ""))
        entry = _entry_point(code)
        if not entry:
            raise SystemExit(f"MBPP row {source_index} has no top-level entry point")
        assertions = [
            str(test).strip()
            for test in row.get("test_list", [])
            if str(test).strip() and _assert_calls_entry(str(test), entry)
        ]
        if len(assertions) < 2:
            raise SystemExit(f"MBPP row {source_index} has fewer than two usable tests")
        assertion_fingerprints = [_assertion_fingerprint(value) for value in assertions]
        if len(set(assertion_fingerprints)) != len(assertion_fingerprints):
            raise SystemExit(f"MBPP row {source_index} contains duplicate assertions")

        setup = "\n".join(str(value) for value in (row.get("test_imports", []) or []))
        namespace: dict = {}
        try:
            if setup:
                exec(setup, namespace)
            exec(code, namespace)
            for assertion in assertions:
                exec(assertion, namespace)
        except Exception as exc:
            raise SystemExit(f"gold MBPP program failed row {source_index}: {exc}") from exc

        problems.append(
            {
                "entry_point": entry,
                "hidden_tests": assertions[1:],
                "prompt": str(row["prompt"]).strip() + f"\n\nWrite a function named `{entry}`.",
                "task_id": f"mbpp/{row['task_id']}",
                "test_setup": setup,
                "visible_tests": assertions[:1],
            }
        )

    problems_path = data_dir / "problems.json"
    problems_path.write_text(_canonical_json(problems))
    manifest = {
        "count": len(problems),
        "dataset": DATASET_ID,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", ""),
        "dataset_revision": DATASET_REVISION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "problems_sha256": _sha256(problems_path),
        "protocol": PROTOCOL,
        "split": "test",
    }
    (data_dir / "manifest.json").write_text(_canonical_json(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    root = args.data_root / "code-generation"
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _stage_model(root / "models")
    manifest = _build_problems(data_dir)
    print(
        "CODEGEN_PREPARE_COMPLETE "
        f"protocol={PROTOCOL} count={manifest['count']} "
        f"problems_sha256={manifest['problems_sha256']} model={model_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
