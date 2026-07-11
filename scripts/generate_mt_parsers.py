#!/usr/bin/env python3
"""Render or verify all machine-translation parser copies deterministically."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TASK = "mt-decoding-beam"
CANONICAL_SURFACE = "build_beam_config"
SURFACES = {
    "mt-batch-maxlen": "build_max_new_tokens",
    "mt-decoding-beam": "build_beam_config",
    "mt-decoding-strategy": "build_strategy",
    "mt-decoding-temperature": "build_temperature",
    "mt-diverse-beam": "build_divbeam_config",
    "mt-early-stopping": "build_early_stopping",
    "mt-length-penalty": "build_length_config",
    "mt-no-repeat-ngram": "build_norep_config",
    "mt-postprocess-detok": "build_postproc",
    "mt-repetition-penalty": "build_reppen_config",
    "mt-sampling-vs-beam": "build_mode",
    "mt-tokenization-truncation": "build_source_max_tokens",
}


def canonical_source() -> str:
    path = ROOT / "tasks" / CANONICAL_TASK / "parser.py"
    source = path.read_text(encoding="utf-8")
    task_line = f'EXPECTED_TASK = "{CANONICAL_TASK}"'
    surface_line = f'EXPECTED_SURFACE = "{CANONICAL_SURFACE}"'
    if source.count(task_line) != 1 or source.count(surface_line) != 1:
        raise ValueError("canonical MT parser identity literals are ambiguous")
    return source


def render(task_name: str, surface_name: str) -> str:
    source = canonical_source()
    source = source.replace(
        f'EXPECTED_TASK = "{CANONICAL_TASK}"',
        f'EXPECTED_TASK = "{task_name}"',
        1,
    )
    source = source.replace(
        f'EXPECTED_SURFACE = "{CANONICAL_SURFACE}"',
        f'EXPECTED_SURFACE = "{surface_name}"',
        1,
    )
    return source


def mismatches() -> list[str]:
    bad = []
    for task_name, surface_name in SURFACES.items():
        path = ROOT / "tasks" / task_name / "parser.py"
        if path.read_text(encoding="utf-8") != render(task_name, surface_name):
            bad.append(task_name)
    return bad


def write_all() -> None:
    for task_name, surface_name in SURFACES.items():
        path = ROOT / "tasks" / task_name / "parser.py"
        path.write_text(render(task_name, surface_name), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_all()
    bad = mismatches()
    if bad:
        raise SystemExit(f"non-canonical MT parser copies: {', '.join(bad)}")
    print(f"MT_PARSERS_CANONICAL count={len(SURFACES)}")


if __name__ == "__main__":
    main()
