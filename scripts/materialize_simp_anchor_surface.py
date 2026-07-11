#!/usr/bin/env python3
"""Materialize one trusted full-split text-simplification anchor surface."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/text-simplification"
SURFACES = {
    "simp-beam-width": ("beamwidth.py", "build_num_beams"),
    "simp-decoding-beam": ("beam.py", "build_beam_config"),
    "simp-decoding-strategy": ("strategy.py", "build_strategy"),
    "simp-decoding-temperature": ("temperature.py", "build_temperature"),
    "simp-input-truncation": ("truncation.py", "build_max_input_tokens"),
    "simp-length-control": ("length.py", "build_length_config"),
    "simp-minlen-floor": ("minlen.py", "build_min_length"),
    "simp-model-capacity": ("capacity.py", "build_model_choice"),
    "simp-nucleus-sampling": ("nucleus.py", "build_top_p"),
    "simp-source-policy": ("policy.py", "build_policy"),
}


def _load_common():
    path = VENDOR / "common.py"
    spec = importlib.util.spec_from_file_location("simp_anchor_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(SURFACES), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    filename, symbol = SURFACES[args.task]
    source = VENDOR / "solution" / filename
    task = ROOT / "tasks" / args.task
    config = json.loads((task / "config.json").read_text())
    baseline = config.get("baselines", {}).get(args.baseline)
    if not isinstance(baseline, dict) or set(baseline) != {"edit_ops"}:
        raise ValueError(f"unknown or malformed baseline {args.task}:{args.baseline}")
    source_rel = config["files"][0]["filename"]
    if Path(source_rel).name != filename:
        raise ValueError(f"configured source mismatch for {args.task}")

    lines = source.read_text().splitlines()
    namespace = runpy.run_path(str(task / baseline["edit_ops"]))
    operations = namespace.get("OPS")
    if not isinstance(operations, list) or not operations:
        raise ValueError(f"{args.task}:{args.baseline} has no edit operations")
    for operation in sorted(operations, key=lambda item: item["start_line"], reverse=True):
        if (not isinstance(operation, dict)
                or operation.get("op") != "replace"
                or operation.get("file") != source_rel
                or not isinstance(operation.get("content"), str)):
            raise ValueError(f"unsafe edit operation in {args.task}:{args.baseline}")
        start = operation.get("start_line")
        end = operation.get("end_line")
        if (not isinstance(start, int) or not isinstance(end, int)
                or not 1 <= start <= end <= len(lines)):
            raise ValueError(f"out-of-range edit in {args.task}:{args.baseline}")
        lines[start - 1:end] = operation["content"].splitlines()

    candidate = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(candidate)
    common = _load_common()
    common.load_surface(str(args.output), symbol)()
    print(f"MATERIALIZED task={args.task} baseline={args.baseline} path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
