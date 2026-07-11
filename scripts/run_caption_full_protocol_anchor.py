#!/usr/bin/env python3
"""Materialize and run one full official caption baseline on one visible GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TASK_MODE = {
    "caption-decoding-strategy": "decoding",
    "caption-visual-mapping": "mapping",
    "caption-training-objective": "objective",
    "caption-feature-prep": "featureprep",
    "caption-mapping-init": "init",
    "caption-train-sampling": "sampling",
    "caption-optimizer": "optimizer",
    "caption-prompt-format": "prompt",
    "caption-feature-augment": "augment",
    "caption-token-weighting": "weighting",
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RESULT = re.compile(
    rf"CAPTION_RESULT protocol=flickr8k_official_v1 mode=(\w+) "
    rf"train_images=(\d+) train_pairs=(\d+) eval_images=(\d+) "
    rf"epochs=(\d+) batch_size=(\d+) steps=(\d+) seed=(\d+) "
    rf"split_sha256=([0-9a-f]{{64}}) manifest_sha256=([0-9a-f]{{64}}) "
    rf"predictions_sha256=([0-9a-f]{{64}}) cider=({NUMBER}) "
    rf"bleu4=({NUMBER}) status=ok"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: str, operation: dict, allowed: dict) -> str:
    if operation.get("op") != "replace":
        raise ValueError("caption anchor accepts exactly one replace operation")
    start, end = int(operation["start_line"]), int(operation["end_line"])
    low, high = int(allowed["start"]), int(allowed["end"])
    if not low <= start <= end <= high:
        raise ValueError("baseline operation is outside the declared edit range")
    lines = source.splitlines()
    if not 1 <= start <= end <= len(lines):
        raise ValueError("baseline operation points outside the solution file")
    return "\n".join(
        lines[: start - 1] + str(operation["content"]).splitlines() + lines[end:]
    ) + "\n"


def parse_completion(output: str, expected_mode: str) -> dict:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    records = [line for line in lines if line.startswith("CAPTION_RESULT")]
    if len(records) != 1 or not lines or records[0] != lines[-1]:
        raise RuntimeError("anchor lacks one final completion record")
    match = RESULT.fullmatch(records[0])
    if match is None:
        raise RuntimeError("anchor completion record is malformed")
    mode, train_images, train_pairs, eval_images, epochs, batch_size, steps, seed, split_hash, manifest_hash, prediction_hash, raw_cider, raw_bleu = match.groups()
    counts = tuple(
        map(int, (train_images, train_pairs, eval_images, epochs, batch_size, steps, seed))
    )
    if mode != expected_mode or counts != (6000, 30000, 1000, 10, 40, 7500, 42):
        raise RuntimeError("anchor completion record binds the wrong protocol")
    cider, bleu = float(raw_cider), float(raw_bleu)
    if not all(math.isfinite(value) for value in (cider, bleu)):
        raise RuntimeError("anchor metrics are non-finite")
    if not 0 <= cider <= 10 or not 0 <= bleu <= 1:
        raise RuntimeError("anchor metrics are outside valid bounds")
    return {
        "mode": mode,
        "cider": cider,
        "bleu4": bleu,
        "split_sha256": split_hash,
        "manifest_sha256": manifest_hash,
        "predictions_sha256": prediction_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_MODE), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    task_dir = root / "tasks" / args.task
    config = json.loads((task_dir / "config.json").read_text())
    if args.baseline not in config["baselines"]:
        raise ValueError(f"unknown baseline {args.task}/{args.baseline}")
    declared = config["files"][0]
    source_path = root / "vendor" / declared["filename"]
    source = source_path.read_text()
    edit_path = task_dir / config["baselines"][args.baseline]["edit_ops"]
    operations = runpy.run_path(str(edit_path))["OPS"]
    if len(operations) != 1 or operations[0].get("file") != declared["filename"]:
        raise ValueError("baseline edit does not match the declared solution")
    candidate = materialize(source, operations[0], declared["edit"][0])

    args.output.mkdir(parents=True, exist_ok=False)
    candidate_path = args.output / source_path.name
    candidate_path.write_text(candidate)
    mode = TASK_MODE[args.task]

    # Validate through the exact trusted static loader before starting a costly run.
    sys.path.insert(0, str(root / "vendor" / "image-captioning"))
    import harness

    literal_config = harness.load_literal_config(candidate_path, mode)
    invocation = [
        sys.executable,
        str(root / "vendor" / "image-captioning" / "harness.py"),
        "--mode",
        mode,
        "--config",
        str(candidate_path),
        "--data-root",
        str(args.data_root / "image-captioning"),
        "--gpt-dir",
        str(args.data_root / "image-captioning" / "gpt2"),
        "--seed",
        "42",
    ]
    protocol = {
        "task": args.task,
        "baseline": args.baseline,
        "mode": mode,
        "literal_config": literal_config,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "harness_sha256": sha256(root / "vendor" / "image-captioning" / "harness.py"),
        "manifest_sha256_before": sha256(
            args.data_root / "image-captioning" / "source_manifest.json"
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    started = time.time()
    output_lines: list[str] = []
    with (args.output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            invocation,
            cwd=root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            output_lines.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    (args.output / "runner.rc").write_text(f"{returncode}\n")
    if returncode != 0:
        raise RuntimeError(f"caption harness exited {returncode}")
    result = parse_completion("".join(output_lines), mode)
    result.update(
        {
            "task": args.task,
            "baseline": args.baseline,
            "returncode": returncode,
            "wall_seconds": time.time() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "candidate_sha256": protocol["candidate_sha256"],
            "harness_sha256": protocol["harness_sha256"],
        }
    )
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"CAPTION_ANCHOR_OK task={args.task} baseline={args.baseline} "
        f"cider={result['cider']:.6f} bleu4={result['bleu4']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
