#!/usr/bin/env python3
"""Render literal task identity into the shared prune-lab parser body."""
from __future__ import annotations

import argparse
from pathlib import Path


TASK_SURFACES = {
    "prune-criterion": "criterion",
    "prune-flops-budget": "flops_budget",
    "prune-layer-budget": "layer_budget",
    "prune-recovery": "recovery",
    "prune-recovery-distill": "recovery_distill",
    "prune-reg-prune": "reg_prune",
    "prune-reinit": "reinit",
    "prune-schedule": "schedule",
    "prune-second-order": "second_order",
    "prune-structured-criterion": "structured_criterion",
    "prune-taylor-estimator": "taylor_estimator",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = PROJECT_ROOT / "tasks"
AUTHORITATIVE = TASKS_ROOT / "prune-criterion" / "parser.py"


def render(source: str, task_id: str, surface: str) -> str:
    task_line = 'EXPECTED_TASK_ID = "prune-criterion"'
    surface_line = 'EXPECTED_SURFACE = "criterion"'
    if source.count(task_line) != 1 or source.count(surface_line) != 1:
        raise SystemExit("authoritative parser identity literals changed unexpectedly")
    return source.replace(
        task_line, f'EXPECTED_TASK_ID = "{task_id}"'
    ).replace(
        surface_line, f'EXPECTED_SURFACE = "{surface}"'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = AUTHORITATIVE.read_text(encoding="utf-8")
    stale = []
    for task_id, surface in TASK_SURFACES.items():
        target = TASKS_ROOT / task_id / "parser.py"
        expected = render(source, task_id, surface)
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != expected:
                stale.append(task_id)
        else:
            target.write_text(expected, encoding="utf-8")
    if stale:
        raise SystemExit("stale prune-lab parsers: " + ", ".join(stale))
    print(f"prune-lab parsers synchronized: {len(TASK_SURFACES)}", flush=True)


if __name__ == "__main__":
    main()
