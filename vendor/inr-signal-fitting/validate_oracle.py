#!/usr/bin/env python3
"""Validate INR baseline edit paths without mutating checked-in solutions.

The default mode applies every active baseline edit in a temporary directory, compiles
the result, loads its JSON configuration in the isolated loader, and compares it with
the measured-candidate manifest. ``--run`` additionally invokes the real harness for
one explicitly selected task, serially. A nonzero harness return code, timeout, missing
metric, duplicate metric, malformed metric, or non-finite value is fatal.
"""
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
import tempfile
from pathlib import Path

import common
import sweep_new_anchors as sweep


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METRIC = re.compile(
    r"INR_METRICS\s+signal=(\S+)\s+psnr=([-\d.eE+]+)\s+"
    r"res=(\d+)\s+elapsed=([-\d.eE+]+)"
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _patched_source(task: str, candidate: str) -> tuple[str, Path]:
    task_dir = ROOT / "tasks" / task
    config = json.loads((task_dir / "config.json").read_text())
    if config.get("_dropped"):
        raise ValueError(f"{task} is retired and cannot be validated as active")
    if candidate not in config.get("baselines", {}):
        raise ValueError(f"{task} has no baseline {candidate!r}")
    if len(config.get("files", [])) != 1:
        raise ValueError(f"{task} must declare exactly one editable file")

    declared = config["files"][0]
    ranges = declared.get("edit", [])
    if len(ranges) != 1:
        raise ValueError(f"{task} must declare exactly one editable range")
    source_path = ROOT / "vendor" / declared["filename"]
    source_lines = source_path.read_text().splitlines()

    edit_path = task_dir / config["baselines"][candidate]["edit_ops"]
    namespace = runpy.run_path(str(edit_path))
    operations = namespace.get("OPS")
    if not isinstance(operations, list) or len(operations) != 1:
        raise ValueError(f"{edit_path} must declare exactly one operation")
    operation = operations[0]
    expected_range = ranges[0]
    if operation.get("op") != "replace":
        raise ValueError(f"{edit_path} operation must be replace")
    if operation.get("file") != declared["filename"]:
        raise ValueError(f"{edit_path} targets the wrong file")
    start = operation.get("start_line")
    end = operation.get("end_line")
    if (start, end) != (expected_range["start"], expected_range["end"]):
        raise ValueError(f"{edit_path} does not replace the complete editable range")
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeError(f"{edit_path} line bounds must be integers")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError(f"{edit_path} line bounds must be integers")
    if start < 1 or end < start or end > len(source_lines):
        raise ValueError(f"{edit_path} line bounds are outside the solution")
    content = operation.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{edit_path} replacement content must be non-empty text")
    patched = "\n".join(
        source_lines[: start - 1] + content.splitlines() + source_lines[end:]
    ) + "\n"
    compile(patched, f"{task}:{candidate}", "exec")
    return patched, source_path


def _parse_metric(output: str, signal: str) -> dict:
    matches = []
    for line in output.splitlines():
        if "INR_METRICS" not in line:
            continue
        match = METRIC.fullmatch(line.strip())
        if match is None:
            raise ValueError("harness emitted a malformed INR_METRICS line")
        matches.append(match)
    if len(matches) != 1:
        raise ValueError(f"harness emitted {len(matches)} INR_METRICS lines; expected one")
    match = matches[0]
    metric_signal = match.group(1)
    psnr = float(match.group(2))
    resolution = int(match.group(3))
    elapsed = float(match.group(4))
    if metric_signal != signal:
        raise ValueError(f"metric signal {metric_signal!r} does not match {signal!r}")
    if resolution != 256:
        raise ValueError(f"unexpected metric resolution {resolution}")
    if not math.isfinite(psnr) or not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("metric values must be finite and elapsed non-negative")
    return {"psnr": psnr, "resolution": resolution, "elapsed": elapsed}


def _run_harness(
    solution: Path, signal: str, seed: int, timeout: float
) -> dict:
    process = subprocess.run(
        [
            sys.executable,
            str(HERE / "harness.py"),
            "--solution",
            str(solution),
            "--signal",
            signal,
            "--seed",
            str(seed),
            "--label",
            signal,
        ],
        cwd=HERE,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    combined = process.stdout + process.stderr
    if process.returncode != 0:
        raise RuntimeError(
            f"harness failed with rc={process.returncode}: {combined[-2000:]}"
        )
    return _parse_metric(combined, signal)


def _validate_candidate(
    task: str,
    candidate: str,
    run_signals: tuple[str, ...],
    seed: int,
    timeout: float,
) -> dict:
    patched, source_path = _patched_source(task, candidate)
    source_before = source_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="inr-validate-") as directory:
        temporary = Path(directory) / source_path.name
        temporary.write_text(patched)
        plan = common.load_surface_config(str(temporary))
        expected = sweep.CANDIDATES[task][candidate]
        if plan != expected:
            raise ValueError(
                f"{task}/{candidate}: edit yields {plan!r}, expected {expected!r}"
            )
        metrics = {
            signal: _run_harness(temporary, signal, seed, timeout)
            for signal in run_signals
        }
    if source_path.read_bytes() != source_before:
        raise RuntimeError(f"validator mutated checked-in solution {source_path}")
    return {
        "config": plan,
        "patched_source_sha256": _sha256_text(patched),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(sweep.SURFACES))
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--signal", action="append", choices=sweep.SIGNALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    tasks = args.task or sorted(sweep.SURFACES)
    if args.candidate and len(tasks) != 1:
        parser.error("--candidate requires exactly one --task")
    if args.run and len(tasks) != 1:
        parser.error("--run requires exactly one --task to prevent accidental fleet runs")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be finite and positive")
    signals = tuple(dict.fromkeys(args.signal or sweep.SIGNALS)) if args.run else ()

    results = {}
    for task in tasks:
        candidates = list(sweep.CANDIDATES[task])
        if args.candidate:
            unknown = set(args.candidate) - set(candidates)
            if unknown:
                parser.error(f"unknown candidates for {task}: {sorted(unknown)}")
            candidates = [name for name in candidates if name in set(args.candidate)]
        task_results = {}
        for candidate in candidates:
            row = _validate_candidate(task, candidate, signals, args.seed, args.timeout)
            task_results[candidate] = row
            print(
                f"VALIDATED task={task} candidate={candidate} "
                f"mode={'run' if args.run else 'static'}",
                flush=True,
            )
        results[task] = task_results

    record = {
        "schema_version": 1,
        "seed": args.seed,
        "signals": list(signals),
        "mode": "run" if args.run else "static",
        "results": results,
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"WROTE {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
