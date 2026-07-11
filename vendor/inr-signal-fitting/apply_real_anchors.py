#!/usr/bin/env python3
"""Read-only consistency check for recorded INR measurements.

Despite the historical filename, this program never writes task files. Only
measurements backed by reviewed raw run evidence may appear in real_anchors.json.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIGNALS = ("low", "medium", "high")
CALIBRATED_TASK = "inr-fourier-frequency"
EXPECTED_EVIDENCE = {
    "terminal_task_id": 95697,
    "terminal_container_id": 4867945,
    "manifest_sha256": "6917c0b86591942653d01eb44ac97c680fd9e5fc7b5892ff1659937012c004f9",
    "anchors_tsv_sha256": "52bed4ed3190943022bf926d9dc85748e65e96223485a23cc159c35e017ac097",
    "terminal_metrics_sha256": "5f3ca6bb31ef47ef7a82df118dee997aad4aff582d4516994ce96d01c2cbe3a8",
}


def _has_score_calls(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(isinstance(node, ast.Call) for node in ast.walk(tree))


def main() -> int:
    recorded = json.loads((HERE / "real_anchors.json").read_text())
    if set(recorded) != {CALIBRATED_TASK}:
        raise ValueError(
            "real_anchors.json must contain only reviewed task-specific frequency "
            f"evidence, got {sorted(recorded)}"
        )

    task_dirs = sorted(path for path in (ROOT / "tasks").glob("inr-*") if path.is_dir())
    checked = 0
    for task_dir in task_dirs:
        config = json.loads((task_dir / "config.json").read_text())
        with (task_dir / "leaderboard.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        if task_dir.name != CALIBRATED_TASK:
            if task_dir.name in recorded:
                raise ValueError(f"{task_dir.name}: unreviewed anchors are recorded")
            if not config.get("_dropped") and config.get("calibration_status") != (
                "pending_exact_zero_task_specific_anchors"
            ):
                raise ValueError(f"{task_dir.name}: active uncalibrated task is not fail-closed")
            if _has_score_calls(task_dir / "score_spec.py"):
                raise ValueError(f"{task_dir.name}: uncalibrated score spec is not empty")
            if rows:
                raise ValueError(f"{task_dir.name}: uncalibrated leaderboard has anchor rows")
            continue

        if config.get("calibration_status") != "measured_task_specific_full_official_anchors":
            raise ValueError(f"{CALIBRATED_TASK}: calibration status is inconsistent")
        if config.get("calibration_evidence") != EXPECTED_EVIDENCE:
            raise ValueError(f"{CALIBRATED_TASK}: evidence identifiers are inconsistent")

        baselines = set(config.get("baselines", {}))
        task_record = recorded[CALIBRATED_TASK]
        if set(task_record) != baselines:
            raise ValueError(f"{CALIBRATED_TASK}: recorded candidate set differs from config")
        baseline_rows = {
            row["model"].removeprefix("baseline:"): row
            for row in rows
            if row.get("model", "").startswith("baseline:")
        }
        if set(baseline_rows) != baselines:
            raise ValueError(f"{CALIBRATED_TASK}: leaderboard candidate set differs from config")
        for baseline in sorted(baselines):
            for signal in SIGNALS:
                measured = float(task_record[baseline][signal])
                leaderboard = float(baseline_rows[baseline][f"psnr_{signal}"])
                if not math.isfinite(measured) or not math.isfinite(leaderboard):
                    raise ValueError(
                        f"{CALIBRATED_TASK}/{baseline}/{signal}: non-finite value"
                    )
                if abs(measured - leaderboard) > 5e-4:
                    raise ValueError(
                        f"{CALIBRATED_TASK}/{baseline}/{signal}: "
                        f"leaderboard={leaderboard} recorded={measured}"
                    )
                checked += 1

    print(
        f"READ_ONLY_ANCHOR_CHECK_OK calibrated_tasks=1 measurements={checked} "
        f"fail_closed_tasks={len(task_dirs) - 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
