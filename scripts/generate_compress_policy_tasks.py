#!/usr/bin/env python3
"""Generate the nine frozen-zoo policy tasks that replace toy training tasks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
PROTOCOL = "compressai_zoo_kodak24_q1q8_policy_v1"
PROTOCOL_SHA = "4b84d6ac0f8af07206b674824608ddbf1ff6e05037f363048521c5869bc525c9"
FAMILIES = ("factorized", "hyperprior_scale", "meanscale")
SETTINGS = ("full", "low", "mid", "high")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    mode: str
    surface_file: str
    surface_name: str
    objective: str
    title: str
    question: str
    mapping: str
    constraint: str | None = None


SPECS = (
    TaskSpec(
        "compress-content-dispatch",
        "content",
        "content_policy.py",
        "content_policy",
        "rd12",
        "Content-aware codec dispatch",
        "Choose an official codec family separately for low-, mid-, and high-texture Kodak strata.",
        "The returned tuple is ordered `(low_texture, mid_texture, high_texture)`.",
    ),
    TaskSpec(
        "compress-quality-dispatch",
        "quality",
        "quality_policy.py",
        "quality_policy",
        "rd12",
        "Quality-band codec dispatch",
        "Choose an official codec family separately for low, middle, and high quality-index bands.",
        "The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.",
    ),
    TaskSpec(
        "compress-low-rate-policy",
        "quality",
        "low_rate_policy.py",
        "low_rate_policy",
        "lowq_rd12",
        "Low-rate codec policy",
        "Choose a family for each quality band while optimizing the low-quality operating points.",
        "The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.",
    ),
    TaskSpec(
        "compress-parameter-budget",
        "global",
        "parameter_budget_policy.py",
        "parameter_budget_policy",
        "rd12",
        "Parameter-budget codec choice",
        "Choose one official codec family for the complete matrix under the task's parameter constraint.",
        "Return one family string.",
        "mean_params",
    ),
    TaskSpec(
        "compress-objective-policy",
        "quality",
        "objective_policy.py",
        "objective_policy",
        "rd6",
        "Distortion-weighted codec policy",
        "Choose a family per quality band for a reconstruction-focused rate-distortion objective.",
        "The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.",
    ),
    TaskSpec(
        "compress-robust-policy",
        "content",
        "robust_policy.py",
        "robust_policy",
        "rd18",
        "Content-robust codec policy",
        "Choose a family per content stratum; scoring requires performance on every stratum and the full set.",
        "The returned tuple is ordered `(low_texture, mid_texture, high_texture)`.",
    ),
    TaskSpec(
        "compress-bitrate-policy",
        "quality",
        "bitrate_policy.py",
        "bitrate_policy",
        "target_utility",
        "Bitrate-schedule codec policy",
        "Choose a family per quality band for a fixed, quality-indexed bitrate schedule.",
        "The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.",
    ),
    TaskSpec(
        "compress-stream-budget",
        "global",
        "stream_budget_policy.py",
        "stream_budget_policy",
        "rd12",
        "Entropy-stream-budget codec choice",
        "Choose one official codec family for the complete matrix under the entropy-stream constraint.",
        "Return one family string.",
        "mean_streams",
    ),
    TaskSpec(
        "compress-high-rate-policy",
        "quality",
        "high_rate_policy.py",
        "high_rate_policy",
        "highq_rd12",
        "High-rate codec policy",
        "Choose a family for each quality band while optimizing the high-quality operating points.",
        "The tuple is ordered `(qualities_1_to_3, qualities_4_to_6, qualities_7_to_8)`.",
    ),
)


OLD_SOLUTION_FILES = (
    "activation.py",
    "attention.py",
    "context_model.py",
    "entropy_model.py",
    "latent_channels.py",
    "network_width.py",
    "normalization.py",
    "quantize.py",
    "rd_control.py",
    "residual_blocks.py",
    "upsampling.py",
)
POLICY_SOLUTION_FILES = tuple(spec.surface_file for spec in SPECS)


def baseline_value(spec: TaskSpec, family: str) -> str:
    if spec.mode == "global":
        return repr(family)
    return repr((family, family, family))


def task_description(spec: TaskSpec) -> str:
    return f"""# {spec.title}

{spec.question}

Edit `compressai/solution/{spec.surface_file}` so `{spec.surface_name}()` returns a literal policy. {spec.mapping}
Every policy entry must be one of `factorized`, `hyperprior_scale`, or `meanscale`, selecting the pinned official CompressAI families `bmshj2018-factorized`, `bmshj2018-hyperprior`, and `mbt2018-mean`.

Verification evaluates the selected codec dispatch with real `compress()` and `decompress()` calls on all 24 Kodak images at qualities 1 through 8. The complete 192-case matrix is required, and scoring includes the full set plus all three fixed content strata. Invalid policies or incomplete output receive zero.
"""


def parser_source(spec: TaskSpec) -> str:
    return f'''"""Strict parser binding for {spec.task_id}."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_shared():
    candidates = (
        Path("/workspace/compressai/policy_parser.py"),
        Path(__file__).resolve().parents[2] / "vendor" / "compressai" / "policy_parser.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise ImportError("strict CompressAI policy parser is unavailable")
    spec = importlib.util.spec_from_file_location("compressai_policy_parser", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError("strict CompressAI policy parser loader is unavailable")
    spec.loader.exec_module(module)
    return module


Parser = _load_shared().make_parser({spec.mode!r}, {spec.surface_name!r})
'''


def run_script(spec: TaskSpec) -> str:
    return f'''#!/bin/bash
set -euo pipefail

cd /workspace/compressai
test -f harness_zoo_entropy.py
test -f harness_zoo_policy.py
test -f policy_parser.py
test -f /data/compressai-zoo/protocol.json
test -d /data/compressai-zoo/kodak
test -d /data/compressai-zoo/checkpoints

/opt/conda/bin/python -I harness_zoo_policy.py \\
  --solution solution/{spec.surface_file} \\
  --surface-name {spec.surface_name} \\
  --mode {spec.mode} \\
  --data-root /data/compressai-zoo/kodak \\
  --checkpoint-root /data/compressai-zoo/checkpoints \\
  --protocol /data/compressai-zoo/protocol.json \\
  --protocol-sha256 {PROTOCOL_SHA}
'''


def placeholder_score_spec(spec: TaskSpec) -> str:
    constraint_lines = ""
    if spec.constraint:
        constraint_lines = f'''    _constraint = f"{spec.constraint}_{{_setting}}"
    term(_constraint, penalty_upper(col(_constraint).higher().id(), target=1.0))
    setting(_setting, weighted_mean((_metric, 1.0)), constraints=[_constraint])
'''
    else:
        constraint_lines = "    setting(_setting, weighted_mean((_metric, 1.0)))\n"
    return f'''"""PENDING_FULL_OFFICIAL: replace only from terminal policy anchors."""
from mlsbench.scoring.dsl import *

for _setting in {SETTINGS!r}:
    _metric = f"{spec.objective}_{{_setting}}"
    term(_metric, col(_metric).higher().id().sigmoid(ref=const(0.0), scale=1.0))
{constraint_lines}
task(gmean("full", "low", "mid", "high"))
'''


def config(spec: TaskSpec) -> dict:
    current = f"compressai/solution/{spec.surface_file}"
    pruned_solutions = [
        f"compressai/solution/{name}"
        for name in (*OLD_SOLUTION_FILES, *POLICY_SOLUTION_FILES)
        if name != spec.surface_file
    ]
    return {
        "allow_create": False,
        "rigorous_codebase": True,
        "calibration_protocol": PROTOCOL,
        "verifier_only_package_files": [
            "compressai/harness_zoo_entropy.py",
            "compressai/harness_zoo_policy.py",
            "compressai/policy_parser.py",
        ],
        "agent_pruned_package_files": [
            "compressai/common.py",
            "compressai/harness_arch.py",
            "compressai/harness_entropy.py",
            "compressai/harness_quant.py",
            "compressai/harness_rd.py",
            "compressai/nets.py",
            *pruned_solutions,
        ],
        "seeds": [42],
        "test_cmds": [
            {
                "cmd": "scripts/full.sh",
                "label": "kodak24_q1q8",
                "score_settings": list(SETTINGS),
                "group": 1,
                "compute": 1.0,
                "time": "6:00:00",
                "mem": 64,
                "package": "compressai",
            }
        ],
        "baselines": {
            f"uniform_{family}": {"edit_ops": f"edits/uniform_{family}.edit.py"}
            for family in FAMILIES
        },
        "files": [
            {
                "filename": current,
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": 6, "end": 6}],
            }
        ],
    }


def edit_source(spec: TaskSpec, family: str) -> str:
    filename = f"compressai/solution/{spec.surface_file}"
    content = f"    return {baseline_value(spec, family)}"
    return f'''"""Uniform {family} policy anchor for {spec.task_id}."""

OPS = [
    {{
        "op": "replace",
        "file": {filename!r},
        "start_line": 6,
        "end_line": 6,
        "content": {content!r},
    }},
]
'''


def generate(spec: TaskSpec) -> None:
    task_dir = TASKS / spec.task_id
    if task_dir.exists():
        raise FileExistsError(f"refusing to overwrite {task_dir}")
    (task_dir / "scripts").mkdir(parents=True)
    (task_dir / "edits").mkdir()
    (task_dir / "config.json").write_text(
        json.dumps(config(spec), indent=2, sort_keys=False) + "\n"
    )
    (task_dir / "task_description.md").write_text(task_description(spec))
    (task_dir / "parser.py").write_text(parser_source(spec))
    (task_dir / "scripts" / "full.sh").write_text(run_script(spec))
    (task_dir / "scripts" / "full.sh").chmod(0o755)
    (task_dir / "score_spec.py").write_text(placeholder_score_spec(spec))
    metric_columns = [f"{spec.objective}_{setting}" for setting in SETTINGS]
    if spec.constraint:
        metric_columns.extend(f"{spec.constraint}_{setting}" for setting in SETTINGS)
    with (task_dir / "leaderboard.csv").open("w", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(
            ["timestamp", "model", "is_final", "seed", *metric_columns, "elapsed"]
        )
    for family in FAMILIES:
        (task_dir / "edits" / f"uniform_{family}.edit.py").write_text(
            edit_source(spec, family)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=[spec.task_id for spec in SPECS])
    args = parser.parse_args()
    wanted = set(args.task or [spec.task_id for spec in SPECS])
    for spec in SPECS:
        if spec.task_id in wanted:
            generate(spec)
            print(spec.task_id)


if __name__ == "__main__":
    main()
